Each task has its own threshold, chosen on its own calibration rows: the lowest (fastest) whose accuracy beats the 27B alone by at least 1%. Test rows below.

## The headline: 8 adapters (every one measured, except emotion)

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

## The inbox agent's five decisions (guard, triage, support-intents, tools, ground)

| Task | rows | 27B alone: accuracy | 27B alone: mean time | Cascade: accuracy | Cascade: mean time | sent to 27B | faster | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mean of the five** | | 87.7% | 7.69 s | 95.7% | 0.25 s | | 39.0x (geometric) | |

## Left out of the headline: emotion

| Task | rows | 27B alone: accuracy | 27B alone: mean time | Cascade: accuracy | Cascade: mean time | sent to 27B | faster | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| emotion | 500 | 35.6% | 4.79 s | 60.6% | 0.11 s | 0.0% | 42.5x | 0.00 |

Left out because picking the single strongest of 27 emotions (or neutral) in short Reddit comments is hard even for people, and the human labels often disagree. Including it, the mean over all 9 measured adapters is 91.4% for the cascade against 80.9% for the 27B alone, so leaving it out makes the headline gain smaller, not larger.

For comparison, one shared threshold for the five agent tasks (best: 0.51); mean; faster = geometric mean of per-task speed-ups:

| X | calibration accuracy | test accuracy | sent to 27B | faster |
|---:|---:|---:|---:|---:|
| 0.00 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.05 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.10 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.15 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.20 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.25 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.30 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.35 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.40 | 96.8% | 95.6% | 0.0% | 44.1x |
| 0.45 | 96.8% | 95.7% | 0.1% | 42.2x |
| 0.50 | 96.8% | 95.7% | 0.6% | 33.0x |
| 0.51 | 96.9% | 95.8% | 0.9% | 30.5x |
| 0.55 | 96.8% | 95.9% | 1.6% | 25.6x |
| 0.60 | 96.7% | 95.9% | 2.4% | 20.5x |
| 0.65 | 96.7% | 95.7% | 3.1% | 18.3x |
| 0.70 | 96.5% | 95.9% | 4.5% | 16.1x |
| 0.75 | 96.1% | 95.9% | 5.7% | 14.0x |
| 0.80 | 95.7% | 95.5% | 7.6% | 11.4x |
| 0.85 | 95.7% | 95.3% | 9.3% | 9.9x |
| 0.90 | 95.5% | 95.2% | 11.7% | 8.2x |
| 0.95 | 95.2% | 94.7% | 15.1% | 6.3x |
| 1.00 | 88.7% | 87.7% | 100.0% | 1.0x |
