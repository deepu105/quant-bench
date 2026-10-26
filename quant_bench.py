#!/usr/bin/env python3
"""Quant x engine benchmark for local OpenAI-compatible servers: speed and hard-problem accuracy,
repeated N times per combination, with mean / best / worst per run.

For every [engines.*] x [models.*] pair in the config: start the server once, run each set N times
(seed 1000 + run), stop the server. Results are one JSON file per request, so an interrupted run
resumes where it stopped.

Sets:
  speed  cold prompt of ~32k tokens + 2 follow-ups on the same conversation (TTFT, prefill, decode)
  math   MathArena problems (default AIME 2026 #15, HMMT Feb 2026 #17 #20), MathArena's prompt,
         graded on the last \\boxed{} answer
  lcb    LiveCodeBench v6 AtCoder hard stdin problems (default ARC 196 C, A, D), LCB's prompt,
         graded on all public + private tests (10 s per test). Runs model-written Python locally.
  agent  Aider polyglot exercises solved by a coding agent (default `pi -p`) against the server,
         graded with the pristine tests. Needs the agent CLI, cargo, go and cmake + g++.

Usage:
  quant_bench.py prep [--data DIR] [--agent]     download problems (and the polyglot repo)
  quant_bench.py run CONFIG [--only ENGINE/MODEL] [--runs N] [--sets speed,math] [--dry-run]
  quant_bench.py summary CONFIG                  markdown tables -> <results>/summary.md

Python 3.11+, standard library only (`tokenizers` is optional, see README).
"""
import argparse
import base64
import csv
import http.client
import io
import json
import os
import pickle
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_DATA = Path("~/.cache/quant-bench").expanduser()
MATH_DEFAULT = ["aime_2026:15", "hmmt_feb_2026:17", "hmmt_feb_2026:20"]
LCB_DEFAULT = ["arc196_c", "arc196_a", "arc196_d"]
LCB_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"
ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=MathArena/{ds}&config=default&split=train&offset={off}&length=100"
POLYGLOT_URL = "https://github.com/Aider-AI/polyglot-benchmark"
AGENT_DEFAULT = ["rust/macros", "rust/xorcism", "rust/decimal", "rust/poker",
                 "go/zebra-puzzle", "go/alphametics", "go/dominoes", "go/connect",
                 "cpp/binary-search-tree", "cpp/bank-account", "cpp/zebra-puzzle"]
AGENT_TEST_CMD = {"rust": "cargo test --offline -- --include-ignored", "go": "go test ./...",
                  "cpp": "cmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=1 && cmake --build build"}
LCB_SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) "
              "and will generate a correct Python program that matches the specification and passes all tests.")
LCB_FORMAT = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
              "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
              "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
              "output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n")
SPEED_TASK = ("Above are {n} competitive programming problems. Pick problem {k}, explain the algorithm "
              "in a few sentences, then give a complete Python solution.")
SPEED_FOLLOWUPS = ["Now write unit tests for that solution, covering the edge cases.",
                   "Review the solution and the tests for bugs or missed cases, and list concrete fixes.",
                   "Rewrite the solution with those fixes applied. Show the full code."]

DEFAULTS = {
    "results": "results", "data": str(DEFAULT_DATA), "runs": 3, "sets": ["speed", "math", "lcb"],
    "host": "127.0.0.1", "port": 8090, "ctx": 65536, "served_name": "bench", "max_tokens": 60000,
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "ready_timeout_s": 1800, "request_timeout_s": 7200, "stop_timeout_s": 120, "min_mem_gib": 4,
    "tokenizer": "",
    "speed": {"prompt_tokens": 32000, "followups": 2, "max_tokens": 16384},
    "math": {"extra_body": {}}, "lcb": {"extra_body": {}, "test_timeout_s": 10},
    "agent": {"tasks": AGENT_DEFAULT, "timeout_s": 1800,
              "cmd": "pi -p --provider bench --model {served_name} --offline "
                     "--tools read,bash,edit,write --session-dir {work}/session {prompt}"},
}


class Stop(Exception):
    """Ends the whole bench (memory watchdog, server failed to stop)."""


# ---------- config ----------

def flatten(tables, prefix=""):
    """TOML reads [models.qwen3-0.6b] as nested tables; join the dotted name back."""
    out = {}
    for k, v in tables.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            out.update(flatten(v, name))
        else:
            out[name] = v
    return out


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))
    raw = tomllib.loads(Path(path).read_text())
    for k, v in raw.items():
        if k in ("speed", "math", "lcb", "agent") and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    base = Path(path).resolve().parent
    cfg["results"] = (base / Path(cfg["results"]).expanduser()).resolve()
    cfg["data"] = Path(cfg["data"]).expanduser()
    cfg["engines"], cfg["models"] = flatten(cfg.get("engines", {})), flatten(cfg.get("models", {}))
    if not cfg["engines"] or not cfg["models"]:
        sys.exit("config needs at least one [engines.<name>] and one [models.<name>]")
    return cfg


def combos(cfg, only):
    for e, eng in cfg["engines"].items():
        for m, mod in cfg["models"].items():
            if eng.get("models") and m not in eng["models"]:
                continue
            if only and not any(o in (e, m, f"{e}/{m}") for o in only):
                continue
            yield e, eng, m, mod


def fill(template, values):
    """Split first, then substitute, so a value with spaces stays one argument."""
    return [tok.format(**values) for tok in shlex.split(template)]


# ---------- HTTP ----------

def stream_chat(cfg, body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    conn = http.client.HTTPConnection(cfg["host"], cfg["port"], timeout=cfg["request_timeout_s"])
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500].decode(errors='replace')}")
    ttft, text, reasoning, last = None, [], [], {}
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or ""
            rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
            if (piece or rpiece) and ttft is None:
                ttft = time.perf_counter() - t0
            text.append(piece)
            reasoning.append(rpiece)
            if ch.get("finish_reason"):
                last["finish_reason"] = ch["finish_reason"]
        for k in ("usage", "timings"):
            if chunk.get(k):
                last[k] = chunk[k]
    wall = time.perf_counter() - t0
    conn.close()
    usage, t = last.get("usage") or {}, last.get("timings") or {}
    ntok = usage.get("completion_tokens") or t.get("predicted_n")
    ptok = usage.get("prompt_tokens") or t.get("prompt_n")
    # Client-side figures, so engines that report no timings compare on the same footing.
    return {"wall_s": wall, "ttft_s": ttft, "prompt_tokens": ptok, "completion_tokens": ntok,
            "prefill_tps": ptok / ttft if ptok and ttft else None,
            "decode_tps": (ntok - 1) / (wall - ttft) if ntok and ttft and ntok > 1 and wall > ttft else None,
            "accept": t["draft_n_accepted"] / t["draft_n"] if t.get("draft_n") else None,
            "finish_reason": last.get("finish_reason"), "server_timings": t,
            "text": "".join(text), "reasoning": "".join(reasoning)}


def post_json(cfg, path, body, timeout=30):
    conn = http.client.HTTPConnection(cfg["host"], cfg["port"], timeout=timeout)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class Counter:
    """Token counts for reasoning vs answer text: the server's /tokenize (llama.cpp and its forks),
    else a local tokenizer.json, else split completion_tokens by character share."""

    def __init__(self, cfg):
        self.cfg, self.mode, self.tok = cfg, None, None

    def count(self, s):
        if not s:
            return 0
        if self.mode in (None, "server"):
            try:
                st, data = post_json(self.cfg, "/tokenize", {"content": s})
                if st == 200:
                    self.mode = "server"
                    return len(json.loads(data)["tokens"])
            except (OSError, ValueError, KeyError):
                pass
            self.mode = "local" if self.cfg["tokenizer"] else "chars"
        if self.mode == "local":
            if self.tok is None:
                from tokenizers import Tokenizer
                self.tok = Tokenizer.from_file(str(Path(self.cfg["tokenizer"]).expanduser()))
            return len(self.tok.encode(s, add_special_tokens=False).ids)
        return None

    def split(self, r):
        think, answer = self.count(r["reasoning"]), self.count(r["text"])
        if think is None or answer is None:
            n, a, b = r["completion_tokens"] or 0, len(r["reasoning"]), len(r["text"])
            think = round(n * a / (a + b)) if a + b else 0
            answer = n - think
        return think, answer, self.mode


# ---------- server ----------

def mem_avail_gib():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return None


class Server:
    def __init__(self, cfg, name, eng, values, log_path):
        self.cfg, self.name, self.eng, self.values, self.log_path = cfg, name, eng, values, log_path
        self.proc, self.low_mem, self.stopping = None, False, threading.Lock()

    def start(self):
        argv = fill(self.eng["cmd"], self.values)
        env = {**os.environ, **{k: str(v).format(**self.values) for k, v in self.eng.get("env", {}).items()}}
        self.log = open(self.log_path, "a")
        self.log.write(f"\n# {time.strftime('%F %T')} {shlex.join(argv)}\n")
        self.log.flush()
        t0 = time.perf_counter()
        # Own session, so Ctrl-C in this terminal does not also signal the server.
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        threading.Thread(target=self._watch_mem, daemon=True).start()
        probe = {"model": self.cfg["served_name"], "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}],
                 "chat_template_kwargs": {"enable_thinking": False}}
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during load (rc {self.proc.returncode}); see {self.log_path}")
            if time.perf_counter() - t0 > self.cfg["ready_timeout_s"]:
                raise RuntimeError(f"server not ready after {self.cfg['ready_timeout_s']} s")
            try:
                if post_json(self.cfg, "/v1/chat/completions", probe, timeout=600)[0] == 200:
                    return time.perf_counter() - t0
            except OSError:
                pass
            time.sleep(2)

    def _watch_mem(self):
        low = 0
        while self.proc and self.proc.poll() is None:
            m = mem_avail_gib()
            low = low + 1 if m is not None and m < self.cfg["min_mem_gib"] else 0
            if low >= 3:
                print(f"WATCHDOG: MemAvailable {m:.1f} GiB for 3 s, stopping the server", flush=True)
                self.low_mem = True
                self.stop()
                return
            time.sleep(1)

    def stop(self):
        # The watchdog thread and the run loop can both call this; only one may signal.
        if not self.proc or not self.stopping.acquire(blocking=False):
            if self.proc:
                self.proc.wait()
            return
        if self.proc.poll() is not None:
            return
        if self.eng.get("stop_cmd"):
            subprocess.run(fill(self.eng["stop_cmd"], self.values), check=False)
        else:
            # One SIGTERM per process in the group. A second signal while a GPU kernel runs can fault the GPU.
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            self.proc.wait(self.cfg["stop_timeout_s"])
        except subprocess.TimeoutExpired:
            raise Stop(f"server pid {self.proc.pid} still running {self.cfg['stop_timeout_s']} s after SIGTERM; "
                       "stop it by hand before the next run")


# ---------- sets ----------

def chat_body(cfg, msgs, seed, max_tokens, extra=None, eng=None):
    return {"model": cfg["served_name"], "messages": msgs, **cfg["sampling"], "seed": seed,
            "max_tokens": max_tokens, "max_completion_tokens": max_tokens,
            **(eng or {}).get("extra_body", {}), **(extra or {})}


def slim(r):
    return {k: v for k, v in r.items() if k not in ("text", "reasoning")}


def run_speed(cfg, eng, out, run, label):
    path = out / "speed.json"
    if path.exists():
        return
    sc = cfg["speed"]
    probs = [json.loads(l) for l in open(cfg["data"] / "speed-corpus.jsonl")]
    # This LaTeX-heavy text is ~2.5 characters per token (Qwen3 tokenizer); prompt_tokens records the real count.
    budget, parts, size = int(sc["prompt_tokens"] * 2.5), [], 0
    for i, p in enumerate(probs, 1):
        part = f"## Problem {i}\n{p}\n"
        if size + len(part) > budget:
            break
        parts.append(part)
        size += len(part)
    seed = 1000 + run
    # Unique first line per run, so no engine can serve the cold turn from a prompt cache.
    msgs = [{"role": "user", "content": f"Session {label}-{run}-{time.time_ns()}\n\n" + "\n".join(parts)
             + "\n" + SPEED_TASK.format(n=len(parts), k=min(3, len(parts)))}]
    turns = []
    for k in range(1 + sc["followups"]):
        r = stream_chat(cfg, chat_body(cfg, msgs, seed, sc["max_tokens"], eng=eng))
        tag = "cold" if k == 0 else f"followup{k}"
        pp = f"prefill {r['prefill_tps'] or 0:.0f} t/s " if k == 0 else ""
        print(f"    speed {tag}: prompt {r['prompt_tokens']} ttft {r['ttft_s'] or 0:.1f}s {pp}"
              f"decode {r['decode_tps'] or 0:.1f} t/s "
              f"gen {r['completion_tokens']}", flush=True)
        turns.append({"turn": tag, **slim(r)})
        if k < sc["followups"]:
            msgs = msgs + [{"role": "assistant", "content": r["text"]},
                           {"role": "user", "content": SPEED_FOLLOWUPS[k % len(SPEED_FOLLOWUPS)]}]
    path.write_text(json.dumps({"turns": turns}, indent=1))


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def grade_math(r, item, _cfg):
    got = last_boxed(r["text"])
    if got is None:
        return False, "no \\boxed{}"
    norm = re.sub(r"\\[,!; ]|\s|\\text\{[^}]*\}|\$|,", "", got)
    return norm == item["answer"], got


def run_test(code, t, timeout):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "main.py"
        f.write_text(code)
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], input=t["input"], capture_output=True,
                               text=True, timeout=timeout, cwd=d)
            return p.returncode == 0 and p.stdout.split() == t["output"].split()
        except subprocess.TimeoutExpired:
            return False


def grade_lcb(r, item, cfg):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", r["text"], re.S)
    if not blocks:
        return False, "no code block"
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda t: run_test(blocks[-1], t, cfg["lcb"]["test_timeout_s"]), item["tests"]))
    return all(ok), f"{sum(ok)}/{len(ok)} tests"


def run_problems(cfg, eng, out, run, which, counter):
    items = [json.loads(l) for l in open(cfg["data"] / f"{which}.jsonl")]
    grade = grade_math if which == "math" else grade_lcb
    out.mkdir(parents=True, exist_ok=True)
    for it in items:
        path = out / (re.sub(r"[^\w.-]", "_", it["id"]) + ".json")
        if path.exists():
            continue
        msgs = [{"role": "user", "content": it["prompt"]}]
        if which == "lcb":
            msgs.insert(0, {"role": "system", "content": LCB_SYSTEM})
        r = stream_chat(cfg, chat_body(cfg, msgs, 1000 + run, cfg["max_tokens"], cfg[which]["extra_body"], eng))
        ok, detail = grade(r, it, cfg)
        think, answer, how = counter.split(r)
        res = {"id": it["id"], "pass": ok, "detail": detail, "think_tokens": think, "answer_tokens": answer,
               "token_count": how, **slim(r), "text": r["text"], "reasoning": r["reasoning"]}
        path.write_text(json.dumps(res, indent=1))
        print(f"    {which} {it['id']}: {'PASS' if ok else 'fail'} ({detail}) think {think} answer {answer} "
              f"wall {r['wall_s']:.0f}s decode {r['decode_tps'] or 0:.1f} t/s {r['finish_reason']}", flush=True)


def pi_agent_dir(cfg, out):
    """A clean Pi agent dir holding only a `bench` provider that points at the server."""
    d = out / "pi-agent"
    d.mkdir(parents=True, exist_ok=True)
    (d / "models.json").write_text(json.dumps({"providers": {"bench": {
        "api": "openai-completions", "baseUrl": f"http://{cfg['host']}:{cfg['port']}/v1", "apiKey": "none",
        "name": "Bench", "models": [{"id": cfg["served_name"], "name": cfg["served_name"],
                                     "contextWindow": cfg["ctx"]}]}}}, indent=1))
    return d


def run_agent(cfg, out, run):
    src_root = cfg["data"] / "polyglot-benchmark"
    if not src_root.exists():
        raise RuntimeError("polyglot-benchmark missing; run `quant_bench.py prep --agent`")
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(pi_agent_dir(cfg, out)), "GOFLAGS": "-mod=mod",
           "GOTOOLCHAIN": "local"}
    for task in cfg["agent"]["tasks"]:
        lang, ex = task.split("/")
        src, work = src_root / lang / "exercises" / "practice" / ex, out / lang / ex
        if (work / "result.json").exists():
            continue
        shutil.rmtree(work, ignore_errors=True)
        # .meta holds the reference solution; the agent never sees it.
        shutil.copytree(src, work, ignore=lambda d, names: [n for n in names if n in (".meta", ".docs")])
        meta = json.loads((src / ".meta" / "config.json").read_text())
        docs = [src / ".docs" / "instructions.md", src / ".docs" / "instructions.append.md"]
        prompt = "".join(p.read_text() for p in docs if p.exists()) + (
            "\n####\n\nUse the above instructions to modify the supplied files: " + " ".join(meta["files"]["solution"])
            + "\nDon't change the names of existing functions or classes, as they may be referenced from other "
              "code like unit tests, etc.\nOnly use standard libraries, don't suggest installing any packages.\n"
              f"Do not modify the test files. Run the tests with: {AGENT_TEST_CMD[lang]}")
        argv = fill(cfg["agent"]["cmd"], {"served_name": cfg["served_name"], "work": str(work), "prompt": prompt,
                                          "base_url": f"http://{cfg['host']}:{cfg['port']}/v1"})
        t0, rc = time.perf_counter(), 0
        with open(work / "agent.log", "w") as log:
            try:
                rc = subprocess.run(argv, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=cfg["agent"]["timeout_s"]).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        wall = time.perf_counter() - t0
        grade = out / "grade" / lang / ex
        shutil.rmtree(grade, ignore_errors=True)
        shutil.copytree(work, grade, ignore=lambda d, names: [n for n in names if d == str(work) and n in
                                                              ("session", "agent.log", "build", "target")])
        for f in meta["files"]["test"]:
            shutil.copy(src / f, grade / f)
        for d in ("tests", "test"):
            if (src / d).is_dir():
                shutil.copytree(src / d, grade / d, dirs_exist_ok=True)
        with open(work / "test.log", "w") as log:
            try:
                ok = subprocess.run(["bash", "-c", AGENT_TEST_CMD[lang]], cwd=grade, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600).returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
        res = {"id": task, "pass": ok, "wall_s": wall, "agent_exit": rc}
        (work / "result.json").write_text(json.dumps(res))
        print(f"    agent {task}: {'PASS' if ok else 'fail'} wall {wall / 60:.1f} min (agent exit {rc})", flush=True)


def todo(cfg, out_combo, sets, runs):
    for run in range(1, runs + 1):
        d = out_combo / f"run{run}"
        for s in sets:
            if s == "speed" and not (d / "speed.json").exists():
                return True
            if s in ("math", "lcb"):
                n = sum(1 for _ in open(cfg["data"] / f"{s}.jsonl"))
                if len(list((d / s).glob("*.json"))) < n:
                    return True
            if s == "agent" and len(list((d / "agent").glob("*/*/result.json"))) < len(cfg["agent"]["tasks"]):
                return True
    return False


def cmd_run(args):
    cfg = load_config(args.config)
    sets = args.sets.split(",") if args.sets else cfg["sets"]
    runs = args.runs or cfg["runs"]
    for s in sets:
        if s not in ("speed", "math", "lcb", "agent"):
            sys.exit(f"unknown set {s}")
        need = {"speed": "speed-corpus.jsonl", "math": "math.jsonl", "lcb": "lcb.jsonl"}.get(s)
        if need and not (cfg["data"] / need).exists():
            sys.exit(f"{cfg['data'] / need} missing; run `quant_bench.py prep --data {cfg['data']}`")
    for e, eng, m, mod in combos(cfg, args.only):
        values = {"port": cfg["port"], "host": cfg["host"], "ctx": cfg["ctx"], "served_name": cfg["served_name"],
                  "engine": e, "model": m, **mod}
        out = cfg["results"] / e / m
        try:
            argv = fill(eng["cmd"], values)
        except KeyError as k:
            print(f"=== {e}/{m}: skipped, the engine cmd uses {k} and [models.{m}] has no such key")
            continue
        if args.dry_run:
            print(f"=== {e}/{m}\n  {shlex.join(argv)}")
            continue
        if not todo(cfg, out, sets, runs):
            print(f"=== {e}/{m}: done ({runs} runs)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== {e}/{m} {time.strftime('%T')}", flush=True)
        srv = Server(cfg, e, eng, values, out / "server.log")
        try:
            load = srv.start()
            print(f"  server ready in {load:.0f} s", flush=True)
            with open(out / "load.jsonl", "a") as f:
                f.write(json.dumps({"time": time.strftime("%F %T"), "load_s": load}) + "\n")
            counter = Counter(cfg)
            for run in range(1, runs + 1):
                d = out / f"run{run}"
                d.mkdir(exist_ok=True)
                print(f"  run {run}/{runs}", flush=True)
                for s in sets:
                    if s == "speed":
                        run_speed(cfg, eng, d, run, f"{e}-{m}")
                    elif s == "agent":
                        run_agent(cfg, d / "agent", run)
                    else:
                        run_problems(cfg, eng, d / s, run, s, counter)
        except (RuntimeError, OSError) as err:
            print(f"  FAILED: {err}", flush=True)
        finally:
            srv.stop()
            if srv.low_mem:
                raise Stop("memory watchdog stopped the server; lower ctx or pick a smaller quant")
    if not args.dry_run:
        cmd_summary(args)


# ---------- summary ----------

def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def mmm(xs, spec="{:.1f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = spec.format(statistics.mean(xs))
    return m if len(xs) == 1 else f"{m} ({spec.format(min(xs))}-{spec.format(max(xs))})"


def cmd_summary(args):
    cfg = load_config(args.config)
    R = cfg["results"]
    lines = [f"# quant-bench summary ({time.strftime('%F %T')})", "",
             "Mean over runs, (min-max) across runs. Decode and prefill are client-side "
             "(stream timing), the same method for every engine.", "",
             "## Speed", "",
             "| Engine | Model | Runs | Load s | Cold prompt | Cold TTFT s | Prefill t/s | Decode t/s | "
             "Follow-up TTFT s | Draft accept |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for combo in sorted(p for p in R.glob("*/*") if p.is_dir()):
        e, m = combo.parent.name, combo.name
        runs = sorted(combo.glob("run*/speed.json"))
        if not runs:
            continue
        sp = [json.loads(p.read_text())["turns"] for p in runs]
        loads = [json.loads(l)["load_s"] for l in open(combo / "load.jsonl")] if (combo / "load.jsonl").exists() else []
        cold = [t[0] for t in sp]
        fu = [mean_or_none(x["ttft_s"] for x in t[1:]) for t in sp]
        lines.append(f"| {e} | {m} | {len(sp)} | {mmm(loads, '{:.0f}')} | {mmm([c['prompt_tokens'] for c in cold], '{:.0f}')} | "
                     f"{mmm([c['ttft_s'] for c in cold])} | {mmm([c['prefill_tps'] for c in cold], '{:.0f}')} | "
                     f"{mmm([mean_or_none(x['decode_tps'] for x in t) for t in sp])} | "
                     f"{mmm(fu)} | {mmm([c['accept'] for c in cold], '{:.0%}')} |")
    lines += ["", "## Accuracy", "",
              "Pass = problems solved per run. Per problem = runs that solved it. Think tokens and decode are "
              "per request; wall is per run.", "",
              "| Engine | Model | Set | Runs | Pass mean | Best | Worst | Stdev | Per problem | Think tokens | "
              "Decode t/s | Wall min/run | Hit cap |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for combo in sorted(p for p in R.glob("*/*") if p.is_dir()):
        e, m = combo.parent.name, combo.name
        for s in ("math", "lcb", "agent"):
            per_run = []
            for d in sorted(combo.glob(f"run*/{s}")):
                rs = [json.loads(p.read_text()) for p in sorted(d.glob("*/*/result.json" if s == "agent" else "*.json"))]
                if rs:
                    per_run.append(rs)
                    for r in rs:
                        rows.append({"engine": e, "model": m, "set": s, "run": d.parent.name, "id": r["id"],
                                     **{k: r.get(k) for k in ("pass", "think_tokens", "answer_tokens", "wall_s",
                                                              "decode_tps", "accept", "finish_reason", "token_count")}})
            if not per_run:
                continue
            n = max(len(rs) for rs in per_run)
            passes = [sum(r["pass"] for r in rs) for rs in per_run]
            ids = sorted({r["id"] for rs in per_run for r in rs})
            prob = ", ".join(f"{i} {sum(r['pass'] for rs in per_run for r in rs if r['id'] == i)}/"
                             f"{sum(1 for rs in per_run for r in rs if r['id'] == i)}" for i in ids)
            flat = [r for rs in per_run for r in rs]
            est = " (est.)" if any(r.get("token_count") == "chars" for r in flat) else ""
            sd = f"{statistics.stdev(passes):.2f}" if len(passes) > 1 else "-"
            lines.append(f"| {e} | {m} | {s} | {len(per_run)} | {statistics.mean(passes):.2f}/{n} | {max(passes)}/{n} | "
                         f"{min(passes)}/{n} | {sd} | {prob} | {mmm([r.get('think_tokens') for r in flat], '{:,.0f}')}{est} | "
                         f"{mmm([r.get('decode_tps') for r in flat])} | {mmm([sum(r['wall_s'] for r in rs) / 60 for rs in per_run])} | "
                         f"{sum(r.get('finish_reason') == 'length' for r in flat)} |")
    text = "\n".join(lines) + "\n"
    R.mkdir(parents=True, exist_ok=True)
    (R / "summary.md").write_text(text)
    with open(R / "rows.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(text)
    print(f"-> {R / 'summary.md'}, {R / 'rows.csv'}")


# ---------- prep ----------

class _NoGlobals(pickle.Unpickler):
    """LCB's private tests are a pickled JSON string; refuse anything that would import code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def get(url, dest=None):
    req = urllib.request.Request(url, headers={"User-Agent": "quant-bench"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if dest is None:
            return r.read()
        tmp = Path(str(dest) + ".part")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(dest)


def cmd_prep(args):
    data = Path(args.data).expanduser()
    data.mkdir(parents=True, exist_ok=True)
    rows = {}
    out = []
    for spec in args.math.split(","):
        ds, idx = spec.split(":")
        if ds not in rows:
            rows[ds] = [r["row"] for r in json.loads(get(ROWS_URL.format(ds=urllib.parse.quote(ds), off=0)))["rows"]]
        row = next(r for r in rows[ds] if str(r["problem_idx"]) == idx)
        # MathArena's instruction lines; AIME adds the answer range.
        head = "Put your final answer within \\boxed{}.\n"
        if ds.startswith("aime"):
            head += "The answer is an integer between 0 and 999 inclusive.\n"
        out.append({"id": f"{ds}#{idx}", "prompt": head + "\n" + row["problem"], "answer": str(row["answer"]).strip()})
    (data / "math.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"math: {len(out)} problems -> {data / 'math.jsonl'}")

    raw = data / "lcb-test6.jsonl"
    if not raw.exists():
        print(f"downloading {LCB_URL} (134 MB)", flush=True)
        get(LCB_URL, raw)
    allrows = [json.loads(l) for l in open(raw)]
    if args.lcb_newest:
        pick = sorted((r for r in allrows if r["platform"] == "atcoder" and r["difficulty"] == "hard"),
                      key=lambda r: r["contest_date"], reverse=True)
        want = []
        for r in pick:
            priv = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(r["private_test_cases"])))).load()
            if len(json.loads(priv)) >= 20:
                want.append(r["question_id"])
            if len(want) == args.lcb_newest:
                break
    else:
        want = args.lcb.split(",")
    by_id = {r["question_id"]: r for r in allrows}
    hard = []
    for qid in want:
        r = by_id[qid]
        priv = json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(r["private_test_cases"])))).load())
        tests = json.loads(r["public_test_cases"]) + priv
        hard.append({"id": qid, "date": r["contest_date"][:10],
                     "prompt": f"### Question:\n{r['question_content']}\n\n{LCB_FORMAT}",
                     "tests": [{"input": t["input"], "output": t["output"]} for t in tests]})
    (data / "lcb.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hard))
    print(f"lcb: {', '.join(want)} -> {data / 'lcb.jsonl'}")

    # Speed prompt filler: older LCB statements, so the speed set never shows the model a graded problem.
    corpus = [r["question_content"] for r in sorted(allrows, key=lambda r: r["contest_date"]) if r["question_id"] not in want]
    (data / "speed-corpus.jsonl").write_text("".join(json.dumps(c) + "\n" for c in corpus[:400]))
    print(f"speed corpus: {min(400, len(corpus))} statements -> {data / 'speed-corpus.jsonl'}")

    if args.agent:
        dest = data / "polyglot-benchmark"
        if not dest.exists():
            subprocess.run(["git", "clone", "--depth", "1", POLYGLOT_URL, str(dest)], check=True)
        print(f"polyglot: {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep", help="download problems")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--math", default=",".join(MATH_DEFAULT), help="MathArena dataset:problem_idx list")
    p.add_argument("--lcb", default=",".join(LCB_DEFAULT), help="LiveCodeBench question ids")
    p.add_argument("--lcb-newest", type=int, default=0,
                   help="instead of --lcb: the N newest AtCoder hard problems with 20+ private tests")
    p.add_argument("--agent", action="store_true", help="also clone the Aider polyglot benchmark")
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("run", help="run the matrix")
    p.add_argument("config")
    p.add_argument("--only", action="append", help="ENGINE, MODEL or ENGINE/MODEL (repeatable)")
    p.add_argument("--runs", type=int)
    p.add_argument("--sets", help="comma list: speed,math,lcb,agent")
    p.add_argument("--dry-run", action="store_true", help="print the server command per combination")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("summary", help="tables over the results dir")
    p.add_argument("config")
    p.set_defaults(fn=cmd_summary)
    args = ap.parse_args()
    # SIGTERM (e.g. from `timeout`) takes the Ctrl-C path, so the server still gets stopped.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        args.fn(args)
    except Stop as err:
        sys.exit(f"STOPPED: {err}")
    except KeyboardInterrupt:
        sys.exit("interrupted; re-run the same command to resume")


if __name__ == "__main__":
    main()
