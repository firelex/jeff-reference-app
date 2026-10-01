# Jeff reference app: a 27B model alone vs. a Jeff cascade, on one Mac

[Jeff](https://github.com/firelex/jeff) is a small open decision model: a 0.8B fine-tune of Qwen3.5
([`mstrasser/Jeff-Qwen3.5-0.8B`](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B), revision `v1.2`) that answers a
multiple-choice decision in one forward pass, with a probability for every option. Nine LoRA adapters,
`mstrasser/Jeff-Qwen3.5-0.8B-<name>`, specialise it: guard, triage, support-intents, tools, ground, nav, emotion, spam
and legal-clauses.

This app compares two ways of making the same decisions on an Apple-silicon Mac:

- **27B alone:** Qwen3.8-27B ([`mlx-community/Qwen3.8-27B-8bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit),
  8-bit MLX weights) answers every query by prompting, with step-by-step reasoning (thinking) off.
- **Cascade:** Jeff with the task's adapter answers first. When Jeff's confidence (the probability of the option it
  chose) is below a per-task threshold, the 27B answers too, and its answer is used. That query's time is Jeff's time
  plus the 27B's.

It reports accuracy and mean time per query for each task. It also has a small demo: a support-inbox agent that runs a
ticket through both setups side by side.

## Result

Measured on an Apple M4 Max with 128 GB of memory. Each task's threshold is chosen on its own calibration rows: the
lowest (fastest) threshold whose accuracy beats the 27B alone by at least 1 point. The numbers below are on separate
test rows. The headline is the mean over every adapter except emotion, which is shown on its own with the reason; the
inbox agent's five decisions are a second figure.

### The headline: 8 adapters (every one measured, except emotion)

| Task | rows | 27B alone: accuracy | 27B alone: mean time | Cascade: accuracy | Cascade: mean time | sent to 27B | faster | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| guard | 300 | 84.0% | 3.92 s | 98.0% | 0.10 s | 0.0% | 38.1x | 0.00 |
| triage | 300 | 81.3% | 3.60 s | 91.0% | 0.06 s | 0.0% | 59.1x | 0.00 |
| support-intents | 300 | 86.0% | 6.44 s | 95.3% | 0.12 s | 0.0% | 54.6x | 0.00 |
| tools | 300 | 90.3% | 11.27 s | 98.0% | 0.31 s | 0.0% | 36.5x | 0.00 |
| ground | 300 | 96.7% | 13.22 s | 96.3% | 0.66 s | 1.7% | 20.1x | 0.64 |
| nav | 300 | 91.3% | 7.21 s | 97.0% | 0.21 s | 0.0% | 34.6x | 0.00 |
| spam | 300 | 88.0% | 2.67 s | 98.7% | 0.07 s | 0.0% | 36.6x | 0.00 |
| legal-clauses | 500 | 75.0% | 16.53 s | 87.8% | 0.47 s | 0.0% | 35.5x | 0.00 |
| **mean of the 8** | | 86.6% | 8.11 s | 95.3% | 0.25 s | | 37.7x (geometric) | |

### The inbox agent's five decisions (guard, triage, support-intents, tools, ground)

| Task | rows | 27B alone: accuracy | 27B alone: mean time | Cascade: accuracy | Cascade: mean time | sent to 27B | faster | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mean of the five** | | 87.7% | 7.69 s | 95.7% | 0.25 s | | 39.0x (geometric) | |

### Left out of the headline: emotion

| Task | rows | 27B alone: accuracy | 27B alone: mean time | Cascade: accuracy | Cascade: mean time | sent to 27B | faster | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| emotion | 500 | 35.6% | 4.79 s | 60.6% | 0.11 s | 0.0% | 42.5x | 0.00 |

Left out because picking the single strongest of 27 emotions (or neutral) in short Reddit comments is hard even for people, and the human labels often disagree. Including it, the mean over all 9 measured adapters is 91.4% for the cascade against 80.9% for the 27B alone, so leaving it out makes the headline gain smaller, not larger.

A threshold of 0.00 means Jeff alone already beats the 27B by at least 1 point on the calibration rows, so no query is
sent to the 27B. The full report, including a comparison with one shared threshold, is
[results/cascade.md](results/cascade.md) ([results/cascade.json](results/cascade.json)).

## How it is measured

- **Rows.** Each task uses a fixed random sample of its held-out test set: 300 rows (500 for emotion and
  legal-clauses), and a separate sample of 150 calibration rows (200 for emotion and legal-clauses). The sample is drawn
  with seed 20260930, so the same rows come out every time, and both setups are scored on exactly the same rows.
- **Same prompt.** The 27B sees Jeff's own prompt (the same input, question, instructions, lettered options and layout,
  from `jeff.model.decision_messages`). Only the answer format differs: it must reply with strict JSON,
  `{"answer": "<letter>"}`. A reply that is not valid JSON or names no option counts as wrong.
- **Scoring.** Accuracy and calibration use the definitions in Jeff's evaluation code (`jeff.evaluate`); see
  `src/jeffref/scoring.py`.
- **Cascade rule.** `src/jeffref/cascade.py`, replayed from the recorded answers and times of both models. Nothing is
  estimated.
- **Memory.** One model is in memory at a time: each benchmark command loads one model and exits, and a lock stops a
  second one from starting beside it. MLX's cache of freed buffers is capped (at 2 GB), because without a cap it grew
  far beyond the model's own size with prompts of many lengths. The 27B needs about **30 GB** of memory; Jeff with one
  adapter about 1.5 GB.

## Rebuild the report from the recorded results

The repository includes everything the published numbers were computed from, so you can rebuild the report without
downloading or running any model:

- `results/big-8bit/<task>-<split>.jsonl`: every answer of the 27B (per-option log-probabilities, reply, time);
- `results/jeff-mlx/<task>-predictions.jsonl` and `<task>.json`: every prediction of Jeff and Jeff + adapter, and
  their metrics;
- `results/samples/<task>.json`: for each sampled row, its id, its options and its correct answer (not its input text),
  plus the SHA-256 of the test file it was drawn from.

```bash
uv sync
uv run jeffref cascade      # rewrites results/cascade.md and results/cascade.json; they come out unchanged
uv run pytest               # includes a check that the report is rebuilt exactly
```

## Reproduce the measurements

You need an Apple-silicon Mac with enough free memory for the 27B (about 30 GB), [uv](https://docs.astral.sh/uv/),
and Python 3.12. Run one benchmark command at a time.

```bash
uv sync

# Download from Hugging Face: Jeff's base model and the adapters with their test sets (all at revision v1.2),
# and the 27B at a pinned commit (about 29 GB).
uv run jeffref fetch --base --tasks all --big

# For each task: the 27B on its test and calibration rows (resumable), then Jeff on the same rows.
for task in guard triage support-intents tools ground nav emotion spam legal-clauses; do
  uv run jeffref bench-27b --task $task
  uv run jeffref bench-jeff --task $task
done

uv run jeffref cascade
```

`fetch` stops with an error if a repository, revision or file is missing, and if a downloaded test or calibration set
is not byte-identical to the one the published results were measured on. `bench-27b` writes one line per answered
row as it goes, so it can be stopped and started again; `--limit N` answers at most N new rows per split, for a quick
try. Your times will differ from ours on other hardware.

| Command | What it does |
|---|---|
| `jeffref fetch [--base] [--tasks a,b \| all] [--big]` | download the base model, adapters with their test sets, the 27B |
| `jeffref bench-27b --task X` | the 27B on task X; writes `results/big-8bit/X-*.jsonl` and `summary.json` |
| `jeffref bench-jeff --task X` | Jeff and Jeff + adapter on task X; writes `results/jeff-mlx/X*.json*` |
| `jeffref cascade` | the report, `results/cascade.md` and `results/cascade.json` |

## The demo: a support-inbox agent

A support agent for a fictional outdoor-gear shop (`src/jeffref/company.py`) handles a ticket with a chain of
decisions, then writes a reply:

1. **guard**: is the text a prompt-injection or jailbreak attempt? (if yes, stop)
2. **triage**: which team, how urgent (1–5), the customer's sentiment (1–5), does a person need to step in?
3. **support-intents**: what exactly the customer wants
4. **tools**: which tool to call, or answer directly, or ask the customer
5. **write the reply** (always the 27B)
6. **ground**: is the drafted reply supported by the retrieved policy excerpts?

In mode A the 27B makes every decision; in mode B Jeff with the matching adapter makes them, and a decision is passed
to the 27B when Jeff's top probability is below a threshold (0.6 by default). The two modes run one after the other, so
each has the whole GPU.

```bash
uv run jeffref fetch --base --tasks guard,triage,support-intents,tools,ground --big
scripts/serve_jeff.sh &         # Jeff's server with every downloaded adapter, on 127.0.0.1:8765
uv run jeffref-demo             # http://127.0.0.1:7871
uv run jeffref-e2e              # optional: time all sample tickets in both modes (results/e2e.json)
```

The ten sample tickets in `data/tickets.json` were written for this demo; you can also paste your own. Here Jeff's
server (about 1.5 GB) runs beside the 27B.

## Layout

| Path | What |
|---|---|
| `src/jeffref/cli.py` | the `jeffref` command: fetch, bench-27b, bench-jeff, cascade |
| `src/jeffref/download.py` | downloads from Hugging Face at explicit revisions |
| `src/jeffref/tasks.py` | the tasks, their sample sizes, the fixed samples and the checks on downloaded test sets |
| `src/jeffref/big.py` | Qwen3.8-27B on MLX: decisions (with per-option probabilities) and replies |
| `src/jeffref/bench.py`, `jeff_bench.py` | accuracy and time of the 27B and of Jeff on the same rows |
| `src/jeffref/scoring.py` | predictions and metrics (Jeff's evaluation definitions) |
| `src/jeffref/cascade.py` | the cascade report |
| `src/jeffref/agent.py`, `company.py`, `runtime.py` | the inbox agent and the fictional shop |
| `src/jeffref/app.py`, `e2e.py`, `serve_jeff.py` | the demo, its end-to-end timing, Jeff's server with MLX's cache capped |
| `results/` | the recorded results and the report |

## Dependencies

Jeff is installed from its public repository at the tag `v1.2`
(`jeff[mac] @ git+https://github.com/firelex/jeff@v1.2`, with its MLX backend). That tag is created when Jeff v1.2
is released; until then `uv sync` cannot resolve it.

## Licence

MIT; see [LICENSE](LICENSE). The models and data sets on Hugging Face have their own licences, stated on their pages.
