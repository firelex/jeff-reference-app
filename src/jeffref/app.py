"""Laptop demo: one support ticket through Mode A (Qwen3.8-27B makes every decision) and Mode B (Jeff + adapters make the
decisions, the 27B only writes the reply), side by side, on this Mac.

    scripts/serve_jeff.sh &          # Jeff server on 127.0.0.1:8765
    jeffref-demo [--threshold 0.6] [--allow-base guard,triage,tools,ground] [--port 7871]

The modes run one after the other (never at the same time), so each gets the whole GPU and the times are comparable."""

import argparse
import html
import os

import gradio as gr

from jeffref.agent import Step, TicketRun, load_tickets, run_ticket
from jeffref.e2e import TICKETS
from jeffref.runtime import footprint_gb, jeff_server_pid, load

STEP_NAMES = {"guard": "1. guard", "triage": "2. triage", "support-intents": "3. intent", "tools": "4. tool",
              "reply": "5. write reply", "ground": "6. ground check"}


def step_rows(run: TicketRun) -> list[list[str]]:
    rows = []
    for s in run.steps:
        if s.step == "reply":
            rows.append([STEP_NAMES["reply"], "reply", "(see below)", "", f"{s.seconds:.2f}", s.answered_by])
            continue
        probability = "" if s.probability is None else f"{s.probability:.2f}"
        by = s.answered_by
        if s.escalated:
            by = f"ESCALATED to 27B (Jeff said '{s.jeff_answer}' at {s.jeff_probability:.2f})"
        rows.append([STEP_NAMES[s.step], s.key, s.label, probability, f"{s.seconds:.2f}", by])
    return rows


def summary(run: TicketRun) -> str:
    escalated = sum(s.escalated for s in run.steps)
    return (f"**Total {run.total_s:.1f} s**: decisions {run.decisions_s:.1f} s, reply {run.reply_s:.1f} s"
            + (f", {escalated} decision(s) escalated to the 27B" if escalated else "") + f"  \n**Outcome:** {run.outcome}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--allow-base", default="")
    parser.add_argument("--port", type=int, default=7871)
    args = parser.parse_args()
    allow_base = [s for s in args.allow_base.split(",") if s]
    tickets = {f"{t['id']}: {t['title']}": t for t in load_tickets(str(TICKETS))}
    big, big_decider, jeff, info = load(args.threshold, allow_base)

    def run_both(choice: str, channel: str, message: str, threshold: float):  # type: ignore[no-untyped-def]
        if not message.strip():
            raise gr.Error("The ticket is empty.")
        jeff.threshold = threshold
        ticket = {"id": "custom" if choice not in tickets else tickets[choice]["id"], "channel": channel,
                  "title": "", "message": message}
        a = run_ticket("A", big_decider, big, ticket)
        b = run_ticket("B", jeff, big, ticket)
        memory = (f"Memory: Qwen3.8-27B ({info['quantization']}) process {footprint_gb(os.getpid()):.1f} GB; "
                  f"Jeff server (base + adapters) {footprint_gb(jeff_server_pid()):.2f} GB")
        speed = f"Mode B took {b.total_s:.1f} s vs {a.total_s:.1f} s for Mode A ({a.total_s / b.total_s:.1f}x faster)."
        return (step_rows(a), summary(a), a.reply, step_rows(b), summary(b), b.reply, memory + "  \n" + speed)

    def pick(choice: str):  # type: ignore[no-untyped-def]
        ticket = tickets[choice]
        return ticket["channel"], ticket["message"]

    adapters = ", ".join(f"{step}: {'adapter' if model == step else 'Jeff base (adapter not trained yet)'}"
                         for step, model in info["adapters"].items())
    headers = ["step", "question", "decision", "probability", "seconds", "answered by"]
    with gr.Blocks(title="Jeff support-agent demo") as demo:
        gr.Markdown("# Support inbox agent: Qwen3.8-27B alone vs Qwen3.8-27B + Jeff\n"
                    f"All on this Mac. 27B: `{info['big_repo']}` via mlx-lm. Jeff v1.2 0.8B via jeff-serve "
                    f"(MLX) with LoRA adapters — {html.escape(adapters)}.")
        with gr.Row():
            choice = gr.Dropdown(list(tickets), label="Sample ticket", value=list(tickets)[0])
            channel = gr.Dropdown(["email", "chat", "phone transcript"], label="Channel", value="email")
            threshold = gr.Slider(0.0, 1.0, value=args.threshold, step=0.05,
                                  label="Mode B escalates a decision to the 27B when Jeff's top probability is below")
        message = gr.Textbox(label="Ticket text (edit freely)", lines=6)
        go = gr.Button("Run both modes", variant="primary")
        memory = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Mode A: Qwen3.8-27B makes every decision")
                a_table = gr.Dataframe(headers=headers, wrap=True)
                a_summary = gr.Markdown()
                a_reply = gr.Textbox(label="Reply (Qwen3.8-27B)", lines=8)
            with gr.Column():
                gr.Markdown("## Mode B: Jeff makes the decisions, the 27B writes")
                b_table = gr.Dataframe(headers=headers, wrap=True)
                b_summary = gr.Markdown()
                b_reply = gr.Textbox(label="Reply (Qwen3.8-27B)", lines=8)
        choice.change(pick, choice, [channel, message])
        demo.load(pick, choice, [channel, message])
        go.click(run_both, [choice, channel, message, threshold],
                 [a_table, a_summary, a_reply, b_table, b_summary, b_reply, memory], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
