"""Shared set-up for the demo app and the end-to-end run: the 27B in this process, Jeff in its own jeff-serve process,
which adapter answers each step, and memory measurements."""

import json
import re
import subprocess
import urllib.request
from typing import Any

from jeffref.agent import BigDecider, JeffDecider
from jeffref.big import BigModel

JEFF_URL = "http://127.0.0.1:8765"
STEPS = ["guard", "triage", "support-intents", "tools", "ground"]


def jeff_models(url: str = JEFF_URL) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as response:
            body = json.load(response)
    except OSError as error:
        raise RuntimeError(f"Jeff server is not reachable at {url}; start it with scripts/serve_jeff.sh") from error
    return [model["name"] for model in body["models"]]


def adapter_map(allow_base: list[str], url: str = JEFF_URL) -> dict[str, str]:
    """step -> model name for Jeff. A step without its adapter is an error unless named in allow_base, in which case
    Jeff's base model answers it and the results say so."""
    served = set(jeff_models(url))
    unknown = set(allow_base) - set(STEPS)
    if unknown:
        raise ValueError(f"--allow-base names unknown steps {sorted(unknown)}; steps are {STEPS}")
    mapping, missing = {}, []
    for step in STEPS:
        if step in served:
            if step in allow_base:
                raise ValueError(f"The {step} adapter is served; remove {step} from --allow-base")
            mapping[step] = step
        elif step in allow_base:
            mapping[step] = "jeff-latest"
        else:
            missing.append(step)
    if missing:
        raise RuntimeError(f"The Jeff server has no adapter for {missing} (served: {sorted(served)}). Download them "
                           f"with jeffref fetch --tasks {','.join(missing)} and restart scripts/serve_jeff.sh, or pass --allow-base {','.join(missing)} to let Jeff's base model "
                           "answer those steps (shown as such in the results).")
    return mapping


def jeff_server_pid(port: int = 8765) -> int:
    result = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True)
    pids = result.stdout.split()
    if len(pids) != 1:
        raise RuntimeError(f"Expected one process listening on port {port}, found {pids}")
    return int(pids[0])


def footprint_gb(pid: int) -> float:
    """The process's physical memory footprint (what Activity Monitor shows as Memory), GPU buffers included."""
    result = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True)
    match = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", result.stdout)
    if result.returncode != 0 or not match:
        raise RuntimeError(f"footprint -p {pid} gave no footprint: {result.stdout[:300]} {result.stderr[:300]}")
    return float(match[1]) * {"KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1.0}[match[2]]


def load(threshold: float, allow_base: list[str]) -> tuple[BigModel, BigDecider, JeffDecider, dict[str, Any]]:
    mapping = adapter_map(allow_base)
    big = BigModel()
    big_decider = BigDecider(big)
    jeff = JeffDecider(JEFF_URL, mapping, big_decider, threshold)
    info = {"adapters": mapping, "quantization": "8bit", "big_repo": big.repo, "big_revision": big.revision}
    return big, big_decider, jeff, info
