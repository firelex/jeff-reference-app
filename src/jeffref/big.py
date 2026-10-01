"""Qwen3.8-27B on this Mac with MLX (mlx-lm): decisions by prompting, and the written reply.

A decision is asked with exactly the prompt Jeff sees (jeff.model.decision_messages: the same state, question,
instructions, lettered options and layout); only the two sentences that say how to answer are changed, from "reply with
the letter code" to "reply with strict JSON {"answer": "<letter code>"}". Thinking is off.

Probabilities: the assistant reply is started with the fixed text {"answer": " (a response prefix), so the next token
is the option code. Every code is one token in the Qwen tokenizer (checked at load), so the model's next-token
distribution restricted to the row's codes, renormalized, is a probability per option. The model then continues
greedily (temperature 0) until the JSON closes; that text is parsed as strict JSON and its "answer" is the decision.
A reply that is not valid JSON, or names a code that is not an option, is recorded as invalid (and scored wrong)."""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jeff.model import decision_messages, options
from jeff.types import DecisionInput

MLX_CACHE_LIMIT = 2 * 2**30  # bytes of freed buffers MLX may keep for reuse
BIG_REPO = "mlx-community/Qwen3.8-27B-8bit"
BIG_REVISION = "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9"  # the commit the published results were measured with
ANSWER_PREFIX = '{"answer": "'
MAX_ANSWER_TOKENS = 8  # enough for the code and the closing "} of the JSON
PREFILL_CHUNK = 2048

JEFF_SYSTEM_END = "Reply with only the selected option code."
JEFF_USER_END = "Return only the letter code of the best option."
SYSTEM_END = 'Reply with strict JSON of the form {"answer": "<option code>"} and nothing else.'
USER_END = 'Return only strict JSON of the form {"answer": "<letter code of the best option>"}, for example {"answer": "A"}.'


def big_model_messages(row: DecisionInput, codes: Sequence[str], layout: str) -> list[dict[str, object]]:
    """Jeff's decision prompt with its answer-format sentences changed to ask for strict JSON."""
    messages = decision_messages(row, codes, layout)
    system, user = messages
    if not isinstance(system["content"], str) or not system["content"].endswith(JEFF_SYSTEM_END):
        raise ValueError("Jeff's system prompt changed; update JEFF_SYSTEM_END in jeffref.big")
    parts = user["content"]
    if not isinstance(parts, list) or len(parts) != 1 or not parts[0]["text"].endswith(JEFF_USER_END):
        raise ValueError("Jeff's user prompt changed (or has images); update JEFF_USER_END in jeffref.big")
    text = parts[0]["text"]
    return [
        {"role": "system", "content": system["content"][: -len(JEFF_SYSTEM_END)] + SYSTEM_END},
        {"role": "user", "content": text[: -len(JEFF_USER_END)] + USER_END},
    ]


def parse_answer(text: str, codes: Sequence[str]) -> str | None:
    """The option code named by a strict-JSON reply, or None when the reply is not exactly {"answer": "<one of codes>"}."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"answer"} or value["answer"] not in codes:
        return None
    return str(value["answer"])


@dataclass
class BigDecision:
    codes: list[str]  # the row's option codes, in option order
    probabilities: list[float]  # per option, renormalized over the codes
    option_log_probs: list[float]  # each code's next-token log-probability over the full vocabulary
    option_mass: float  # the next-token probability that fell on the row's codes before renormalizing
    reply: str  # the full JSON reply text, prefix included
    answer_code: str | None  # parsed from the reply; None when the reply is invalid
    input_tokens: int
    output_tokens: int
    prefill_s: float
    total_s: float


@dataclass
class BigModel:
    path: Path = field(init=False)
    codes: list[str] = field(init=False)

    def __post_init__(self) -> None:
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from mlx_lm import load

        self.mx = mx
        # MLX keeps freed buffers for reuse with no limit by default; with prompts of many lengths that cache grew the
        # benchmark to 96 GB (the model itself is about 30 GB at 8-bit). Cap it.
        mx.set_cache_limit(MLX_CACHE_LIMIT)
        self.repo, self.revision = BIG_REPO, BIG_REVISION
        try:
            self.path = Path(snapshot_download(self.repo, revision=self.revision, local_files_only=True))
        except LocalEntryNotFoundError as error:
            raise RuntimeError(f"{self.repo} at revision {self.revision} is not downloaded; run: jeffref fetch --big") from error
        started = time.perf_counter()
        self.model, self.tokenizer = load(str(self.path))
        self.load_s = time.perf_counter() - started
        # Jeff's 255 answer codes (A, B, ..., Z, AA, ...), each one token in this tokenizer.
        from jeffref.tasks import JEFF_CHECKPOINT
        config = JEFF_CHECKPOINT / "decision_config.json"
        if not config.exists():
            raise FileNotFoundError(f"{config} is missing; download Jeff's base model first: jeffref fetch --base")
        decision = json.loads(config.read_text())
        self.codes = list(decision["codes"])
        self.code_ids: dict[str, int] = {}
        prefix_ids = self.encode(ANSWER_PREFIX)
        for code in self.codes:
            ids = self.encode(ANSWER_PREFIX + code + '"}')
            if ids[: len(prefix_ids)] != prefix_ids or ids[len(prefix_ids)] != self.encode(code)[0] or len(self.encode(code)) != 1:
                raise ValueError(f"Option code {code!r} is not one token after the JSON prefix in {self.repo}")
            self.code_ids[code] = self.encode(code)[0]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def chat_ids(self, messages: list[dict[str, object]], prefix: str = "") -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.encode(text + prefix)

    def prefill(self, ids: list[int]):  # type: ignore[no-untyped-def]
        """Run the prompt through the model in chunks; returns (cache, logits of the last position)."""
        from mlx_lm.models.cache import make_prompt_cache

        mx = self.mx
        cache = make_prompt_cache(self.model)
        for start in range(0, len(ids) - 1, PREFILL_CHUNK):
            chunk = ids[start: min(start + PREFILL_CHUNK, len(ids) - 1)]
            self.model(mx.array([chunk]), cache=cache)
            mx.eval([c.state for c in cache])
        logits = self.model(mx.array([ids[-1:]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        return cache, logits

    def decide(self, row: DecisionInput, layout: str) -> BigDecision:
        mx = self.mx
        started = time.perf_counter()
        count = len(options(row["question"])[0])
        codes = self.codes[:count]
        ids = self.chat_ids(big_model_messages(row, self.codes, layout), ANSWER_PREFIX)
        cache, logits = self.prefill(ids)
        prefill_s = time.perf_counter() - started
        log_probs = logits - mx.logsumexp(logits)
        option_log_probs = log_probs[mx.array([self.code_ids[code] for code in codes])]
        mx.eval(option_log_probs)
        option_mass = float(mx.exp(option_log_probs).sum())
        probabilities = mx.softmax(option_log_probs, axis=-1).tolist()
        # Greedy continuation until the JSON closes, from the unrestricted next-token distribution.
        generated: list[int] = []
        token = int(mx.argmax(logits))
        while True:
            generated.append(token)
            text = self.tokenizer.decode(generated)
            if "}" in text or token in self.tokenizer.eos_token_ids or len(generated) >= MAX_ANSWER_TOKENS:
                break
            logits = self.model(mx.array([[token]]), cache=cache)[0, -1]
            token = int(mx.argmax(logits))
        text = self.tokenizer.decode([t for t in generated if t not in self.tokenizer.eos_token_ids])
        reply = ANSWER_PREFIX + text
        return BigDecision(
            codes=codes, probabilities=[float(p) for p in probabilities],
            option_log_probs=[float(v) for v in option_log_probs.tolist()], option_mass=option_mass, reply=reply,
            answer_code=parse_answer(reply.strip(), codes), input_tokens=len(ids), output_tokens=len(generated),
            prefill_s=prefill_s, total_s=time.perf_counter() - started,
        )

    def write(self, messages: list[dict[str, object]], max_tokens: int = 300) -> tuple[str, int, int, float]:
        """A written reply (thinking off, greedy). Returns (text, input tokens, output tokens, seconds)."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        started = time.perf_counter()
        ids = self.chat_ids(messages)
        text = generate(self.model, self.tokenizer, prompt=ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))
        output_tokens = len(self.encode(text))
        return text.strip(), len(ids), output_tokens, time.perf_counter() - started
