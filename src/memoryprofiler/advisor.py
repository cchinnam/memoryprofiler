"""Rule-based advisor: turns a MemoryProfile into ranked recommendations.

This is the deterministic baseline. Stage 3 swaps `advise()` for an
LLM-backed version (Nemotron via NIM) behind the *same* input/output
contract, so nothing downstream changes.
"""
from __future__ import annotations

from typing import Optional

from .models import MemoryProfile, ProfileReport, Recommendation


def advise(
    profile: MemoryProfile,
    measured_int8: Optional[MemoryProfile] = None,
) -> ProfileReport:
    """Diagnose a profile and produce ranked recommendations.

    If `measured_int8` is supplied, quantization savings are reported from
    real measurement; otherwise they are estimated (~50% of weight bytes).
    """
    w, peak = profile.weights_gb, profile.peak_total_gb
    kv = profile.kv_cache_plus_activations_gb
    wshare = 100.0 * w / peak if peak else 0.0
    recs: list[Recommendation] = []

    # 1. Quantization — the highest-leverage lever when weights dominate.
    if wshare >= 50:
        if measured_int8 is not None:
            nw, npeak = measured_int8.weights_gb, measured_int8.peak_total_gb
            saved = w - nw
            pct = 100.0 * saved / w
            rationale = (
                f"Weights are {wshare:.0f}% of peak. Measured INT8: "
                f"{w:.2f}->{nw:.2f} GB ({pct:.0f}% smaller); peak "
                f"{peak:.2f}->{npeak:.2f} GB, freeing {saved:.2f} GB."
            )
            savings = f"{saved:.2f} GB freed ({pct:.0f}% of weights, measured)"
        else:
            est = w / 2.0
            rationale = (
                f"Weights are {wshare:.0f}% of peak. INT8 would cut them to "
                f"~{est:.2f} GB (~50%, estimated)."
            )
            savings = f"~{w - est:.2f} GB (estimated)"
        recs.append(
            Recommendation(
                priority=1,
                action="Quantize weights FP16 -> INT8",
                rationale=rationale,
                expected_savings=savings,
                quality_cost="~<1% (bitsandbytes / AWQ / GPTQ)",
            )
        )

    # 2. Bottleneck-aware guidance.
    if profile.memory_bound:
        recs.append(
            Recommendation(
                priority=2,
                action="Optimize for memory bandwidth (decode is memory-bound)",
                rationale=(
                    "Each output token reads all weights from HBM, so bandwidth "
                    "dominates. Smaller weights (quantization) and faster HBM help "
                    "more than added FLOPs."
                ),
            )
        )
    else:
        recs.append(
            Recommendation(
                priority=2,
                action="Compute-bound — raise batch size for throughput",
                rationale=(
                    "Compute, not bandwidth, is the limit here. Larger batches "
                    "amortize the work and raise tokens/sec."
                ),
            )
        )

    # 3. KV-cache sizing.
    recs.append(
        Recommendation(
            priority=3,
            action="Re-profile at production context length",
            rationale=(
                f"KV cache + activations here is only {kv:.2f} GB (short test). "
                "KV cache grows ~linearly with context length x concurrency — "
                "size it at your real workload to find the true ceiling."
            ),
        )
    )

    verdict = (
        ("memory-bound" if profile.memory_bound else "compute-bound")
        + f"; weights are {wshare:.0f}% of peak total"
    )

    if measured_int8 is not None:
        saved = w - measured_int8.weights_gb
        headline = (
            f"INT8 cuts weights {100 * saved / w:.0f}% "
            f"({w:.2f}->{measured_int8.weights_gb:.2f} GB), freeing "
            f"{peak - measured_int8.peak_total_gb:.2f} GB on {profile.gpu}"
        )
    else:
        headline = (
            f"{profile.model} uses {peak:.2f} GB on {profile.gpu}; "
            f"weights are {wshare:.0f}% of that"
        )

    recs.sort(key=lambda r: r.priority)
    return ProfileReport(
        profile=profile, verdict=verdict, headline=headline, recommendations=recs
    )


def render_text(report: ProfileReport) -> str:
    """Render a report as a readable plain-text block."""
    p = report.profile
    lines = [
        "=" * 60,
        "              MemoryProfiler — Report",
        "=" * 60,
        f"Model : {p.model}",
        f"GPU   : {p.gpu}   ({p.dtype})",
        "",
        "Memory breakdown:",
        f"  weights            {p.weights_gb:6.2f} GB",
        f"  KV cache + activ.  {p.kv_cache_plus_activations_gb:6.2f} GB",
        f"  peak total         {p.peak_total_gb:6.2f} GB",
        "",
        f"Verdict : {report.verdict}",
        f"Headline: {report.headline}",
        "",
        "Recommendations (ranked):",
    ]
    for r in report.recommendations:
        lines.append(f"  [{r.priority}] {r.action}")
        lines.append(f"      why : {r.rationale}")
        if r.expected_savings:
            lines.append(f"      save: {r.expected_savings}")
        if r.quality_cost:
            lines.append(f"      cost: {r.quality_cost}")
    return "\n".join(lines)
