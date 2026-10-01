"""End-to-end time per ticket, both modes, on the sample tickets; and the memory each part uses.

    scripts/serve_jeff.sh &                      # Jeff server (MLX, adapters)
    jeffref-e2e --threshold 0.6 [--allow-base guard,triage,tools,ground]

Each ticket runs in Mode A (27B only) and Mode B (27B + Jeff), one after the other, after one untimed warm-up ticket
per mode. Writes results/e2e.json and results/e2e-runs.jsonl (every decision of every run)."""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

from jeffref.agent import TicketRun, load_tickets, run_ticket
from jeffref.runtime import footprint_gb, jeff_server_pid, load
from jeffref.scoring import percentile
from jeffref.tasks import ROOT

TICKETS = ROOT / "data" / "tickets.json"
OUT = ROOT / "results"


def summarize(runs: list[TicketRun]) -> dict[str, Any]:
    totals = [r.total_s for r in runs]
    decision_steps = [s for r in runs for s in r.steps if s.step != "reply"]
    return {
        "ticket_s_median": median(totals), "ticket_s_p95": percentile(totals, 0.95), "ticket_s_mean": sum(totals) / len(totals),
        "decisions_s_median": median(r.decisions_s for r in runs),
        "reply_s_median": median(r.reply_s for r in runs if r.reply_s > 0),
        "decision_ms_median": 1000 * median(s.seconds for s in decision_steps),
        "decision_ms_p95": 1000 * percentile([s.seconds for s in decision_steps], 0.95),
        "decisions": len(decision_steps), "escalated": sum(s.escalated for s in decision_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--allow-base", default="", help="steps Jeff's base may answer while their adapter is missing")
    args = parser.parse_args()
    allow_base = [s for s in args.allow_base.split(",") if s]
    tickets = load_tickets(str(TICKETS))
    big, big_decider, jeff, info = load(args.threshold, allow_base)
    import mlx.core as mx

    warm = tickets[0]
    run_ticket("A", big_decider, big, warm)
    run_ticket("B", jeff, big, warm)
    runs: dict[str, list[TicketRun]] = {"A": [], "B": []}
    OUT.mkdir(exist_ok=True)
    with (OUT / "e2e-runs.jsonl").open("w") as out:
        for ticket in tickets:
            for mode, decider in (("A", big_decider), ("B", jeff)):
                run = run_ticket(mode, decider, big, ticket)
                runs[mode].append(run)
                out.write(json.dumps(run.to_json()) + "\n")
                print(f"{ticket['id']:24s} mode {mode}: {run.total_s:6.1f} s (decisions {run.decisions_s:5.1f} s, reply "
                      f"{run.reply_s:5.1f} s)  {run.outcome}", flush=True)
    agreement = []
    for a, b in zip(runs["A"], runs["B"], strict=True):
        a_steps = {s.key: s.answer for s in a.steps if s.step not in ("reply", "ground")}
        b_steps = {s.key: s.answer for s in b.steps if s.step not in ("reply", "ground")}
        shared = set(a_steps) & set(b_steps)
        agreement.append({"ticket": a.ticket_id, "same": sum(a_steps[k] == b_steps[k] for k in shared), "of": len(shared),
                          "differ": {k: [a_steps[k], b_steps[k]] for k in shared if a_steps[k] != b_steps[k]}})
    jeff_pid = jeff_server_pid()
    result = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tickets": len(tickets), "threshold": args.threshold,
        "setup": info, "warm_up_ticket": warm["id"],
        "modes": {"A": summarize(runs["A"]), "B": summarize(runs["B"])},
        "jeff_memory_gb": footprint_gb(jeff_pid),
        "big_process_memory_gb": footprint_gb(os.getpid()),
        "big_mlx_active_gb": mx.get_active_memory() / 2**30, "big_mlx_peak_gb": mx.get_peak_memory() / 2**30,
        "agreement": agreement,
    }
    (OUT / "e2e.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("modes", "jeff_memory_gb", "big_process_memory_gb")}, indent=1))


if __name__ == "__main__":
    main()
