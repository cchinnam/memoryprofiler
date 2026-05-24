"""GPU profiling — runs on a CUDA machine (Colab / Modal / a GPU box).

`torch` and `transformers` are imported *lazily* inside `profile_model`, so
the rest of the package (models, advisor, CLI `advise`) works on a laptop
with no GPU and no torch installed.
"""
from __future__ import annotations

import subprocess
import threading
import time

from .models import MemoryProfile


def _gpu_name() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        return r.stdout.strip().split("\n")[0] or "unknown"
    except Exception:
        return "unknown"


def profile_model(
    model_name: str,
    prompt: str = "Explain GPU memory bandwidth in one paragraph.",
    max_new_tokens: int = 500,
    dtype: str = "float16",
    load_in_8bit: bool = False,
) -> MemoryProfile:
    """Load a model, run one generation, and measure GPU memory + utilization."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    def alloc() -> float:
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e9

    def peak() -> float:
        return torch.cuda.max_memory_allocated() / 1e9

    samples: list[tuple[int, int]] = []
    stop = {"v": False}

    def monitor() -> None:
        while not stop["v"]:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            try:
                g, m = (int(x) for x in r.stdout.strip().split(",")[:2])
                samples.append((g, m))
            except Exception:
                pass
            time.sleep(0.1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    a0 = alloc()

    tok = AutoTokenizer.from_pretrained(model_name)
    load_kwargs: dict = {"device_map": "cuda"}
    if load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        dtype = "int8"
    else:
        load_kwargs["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()
    weights_gb = alloc() - a0

    inp = tok(prompt, return_tensors="pt").to("cuda")
    prompt_tokens = int(inp.input_ids.shape[1])
    torch.cuda.reset_peak_memory_stats()
    a2 = alloc()

    t = threading.Thread(target=monitor)
    t.start()
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    stop["v"] = True
    t.join(timeout=2)

    kv_act = peak() - a2
    peak_total = peak()
    ag = sum(s[0] for s in samples) / len(samples) if samples else 0.0
    am = sum(s[1] for s in samples) / len(samples) if samples else 0.0

    return MemoryProfile(
        model=model_name,
        gpu=_gpu_name(),
        dtype=dtype,
        weights_gb=round(weights_gb, 3),
        kv_cache_plus_activations_gb=round(kv_act, 3),
        peak_total_gb=round(peak_total, 3),
        avg_compute_util_pct=round(ag, 1),
        avg_memory_bw_util_pct=round(am, 1),
        memory_bound=am > ag,
        prompt_tokens=prompt_tokens,
        generated_tokens=int(out.shape[1]) - prompt_tokens,
    )
