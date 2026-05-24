"""LLM-backed advisor — same contract as ``advisor.advise()``, powered by
NVIDIA Nemotron via the NIM OpenAI-compatible endpoint (build.nvidia.com).

If ``NVIDIA_API_KEY`` is unset, the ``openai`` package is missing, or the
call/parse fails, it falls back to the deterministic rule-based advisor and
prints a note to stderr — so you always get a valid ProfileReport.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

from .advisor import advise as rule_based_advise
from .models import MemoryProfile, ProfileReport, Recommendation

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

SYSTEM_PROMPT = (
    "You are a senior GPU performance engineer who optimizes LLM inference "
    "memory. Given a GPU memory profile, produce concise, quantified, ranked "
    "recommendations. Reason about weights vs KV cache vs activations, "
    "memory-bound vs compute-bound decode, quantization (FP16/INT8/FP8/INT4), "
    "GQA, PagedAttention, and batching. Never invent numbers the profile does "
    "not support. Respond with a single JSON object only."
)


def _user_prompt(profile: MemoryProfile, measured_int8: Optional[MemoryProfile]) -> str:
    extra = ""
    if measured_int8 is not None:
        extra = (
            "\n\nMeasured INT8 profile (use for exact savings):\n"
            + measured_int8.model_dump_json(indent=2)
        )
    return (
        f"GPU memory profile:\n{profile.model_dump_json(indent=2)}{extra}\n\n"
        "Return ONLY a JSON object with this shape:\n"
        "{\n"
        '  "verdict": "<one line: memory-bound or compute-bound + key fact>",\n'
        '  "headline": "<one punchy line a customer would quote>",\n'
        '  "recommendations": [\n'
        '    {"priority": 1, "action": "...", "rationale": "...", '
        '"expected_savings": "... or null", "quality_cost": "... or null"}\n'
        "  ]\n"
        "}\n"
        "Provide 2-4 recommendations, priority 1 = highest impact."
    )


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response (handles <think> and fences)."""
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object in model response")


def advise_llm(
    profile: MemoryProfile,
    measured_int8: Optional[MemoryProfile] = None,
    model: Optional[str] = None,
) -> ProfileReport:
    """LLM advisor with graceful fallback to the rule-based advisor."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print(
            "[advisor] NVIDIA_API_KEY not set — using rule-based advisor.",
            file=sys.stderr,
        )
        return rule_based_advise(profile, measured_int8)

    model = model or os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL)
    try:
        from openai import OpenAI

        client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(profile, measured_int8)},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        data = _extract_json(resp.choices[0].message.content)
        recs = [Recommendation(**r) for r in data["recommendations"]]
        recs.sort(key=lambda r: r.priority)
        print(f"[advisor] Nemotron ({model}) via NIM.", file=sys.stderr)
        return ProfileReport(
            profile=profile,
            verdict=data["verdict"],
            headline=data["headline"],
            recommendations=recs,
        )
    except Exception as exc:  # network / auth / bad model / parse — degrade gracefully
        print(
            f"[advisor] LLM call failed ({exc}); using rule-based advisor.",
            file=sys.stderr,
        )
        return rule_based_advise(profile, measured_int8)
