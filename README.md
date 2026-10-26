# quant-bench

Speed and hard-problem accuracy for quants x engines, with every set repeated N times so the
summary shows mean, best and worst run. Any server with an OpenAI-compatible
`/v1/chat/completions` works: llama.cpp and its forks, Halogen, gufo, Strata, and so on.
One file, Python 3.11+, standard library only.

I wrote this for my Swift 1.5 Qwen3.8 Flash-Next benchmark (Oct 2026) and made it standalone. The default
problems are the ones that round used.

## Problems

| Set | Problems | Graded by |
| --- | --- | --- |
| `speed` | ~32k-token cold prompt (older LiveCodeBench statements), then 2 follow-ups on the same conversation | TTFT, prefill t/s, decode t/s, follow-up TTFT, draft acceptance if the server reports it |
| `math` | [MathArena](https://matharena.ai): AIME 2026 #15, HMMT Feb 2026 #17, #20 (the hardest 3 of the 12 picked for that round by MathArena solve rate: 19%, 37%, 21%) | last `\boxed{}` equals the answer |
| `lcb` | [LiveCodeBench](https://livecodebench.github.io) v6 AtCoder hard: ARC 196 C, A, D (2025-04-06, 42-43 tests each) | all public + private tests pass, 10 s per test |
| `agent` | [Aider polyglot](https://github.com/Aider-AI/polyglot-benchmark), Exercism difficulty 7-10: Rust macros, xorcism, decimal, poker; Go zebra-puzzle, alphametics, dominoes, connect; C++ binary-search-tree, bank-account, zebra-puzzle | the exercise's own tests, restored from the pristine copy |

- math and lcb: one request each, the server's default thinking, sampling from the config
  (default temperature 1.0, top_p 0.95, top_k 20), 60,000 output tokens max.
- The LCB problems predate Qwen3.8, so contamination is possible. Pick others with
  `prep --lcb <ids>` or `prep --lcb-newest N`, and other math problems with
  `prep --math aime_2026:1,hmmt_feb_2026:5`.
- With 3 problems per set, only large gaps between quants show. More runs narrow it.

## Use

```bash
python3 quant_bench.py prep                 # ~134 MB download into ~/.cache/quant-bench
python3 quant_bench.py prep --agent         # also clones the polyglot repo for the agent set
cp example.toml my.toml                     # set engines and model paths
python3 quant_bench.py run my.toml --dry-run   # print the server command per combination
python3 quant_bench.py run my.toml
python3 quant_bench.py run my.toml --only llama/swift-iq3_xxs --runs 5 --sets math,lcb
python3 quant_bench.py summary my.toml      # tables -> results/summary.md, all rows -> results/rows.csv
```

- The script starts each engine/model server itself, waits until it answers, runs every set
  `runs` times, then sends one SIGTERM and waits for the exit (`stop_cmd` instead, for docker).
- Re-running the same command resumes: every request is saved as its own JSON file and skipped
  next time. Raise `runs` later and only the new runs are done.
- Each run uses seed `1000 + run`, so run 1 of every quant uses the same seed.
- Placeholders in `cmd`: `{path}`, `{port}`, `{ctx}`, `{served_name}` and any key of the model
  table, e.g. `{draft}` for an MTP head. See `example.toml`.
- On Linux the server is stopped if MemAvailable stays under `min_mem_gib` (4) for 3 s, and the
  bench ends.

## Output

`results/summary.md` has two tables: speed (mean and min-max across runs) and accuracy per set
(pass mean, best and worst run, stdev, passes per problem across runs, thinking tokens, decode
t/s, wall time per run, requests that hit the token cap). `results/rows.csv` has one row per
request for your own plots.

Decode and prefill t/s are measured on the client from stream timing, the same way for every
engine. Thinking vs answer tokens come from the server's `/tokenize` (llama.cpp and forks), else
`tokenizer` in the config (`pip install tokenizers`), else completion tokens split by character
share, marked `(est.)` in the table.

## Agent set

Default agent is [Pi](https://github.com/badlogic/pi-mono) (`pi -p`); the script writes a clean
`PI_CODING_AGENT_DIR` per run with only a `bench` provider pointing at the server. Any other CLI
agent works through `[agent] cmd` with `{prompt}`, `{work}`, `{base_url}`, `{served_name}`.
Needs `cargo`, `go`, `cmake` and a C++ compiler. 30 min per task by default.

## Notes

- The `lcb` and `agent` sets run model-written code on your machine, without a sandbox.
- The speed corpus holds about 100k tokens, so `speed.prompt_tokens` above ~95,000 stops growing.
- LiveCodeBench ships private tests as pickled data; `prep` reads them with an unpickler that
  refuses to import anything.
