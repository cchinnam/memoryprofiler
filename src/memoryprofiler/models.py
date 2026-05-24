"""Typed schemas for MemoryProfiler.

Pure Pydantic — no torch, no GPU. Safe to import anywhere (laptop, server,
the advisor path). The profiler produces a MemoryProfile; the advisor turns
it into a ProfileReport.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MemoryProfile(BaseModel):
    """One profiling run: a model at one precision on one GPU."""

    model: str
    gpu: str
    dtype: str
    weights_gb: float
    kv_cache_plus_activations_gb: float
    peak_total_gb: float
    avg_compute_util_pct: Optional[float] = None
    avg_memory_bw_util_pct: Optional[float] = None
    memory_bound: Optional[bool] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None


class Recommendation(BaseModel):
    priority: int
    action: str
    rationale: str
    expected_savings: Optional[str] = None
    quality_cost: Optional[str] = None


class ProfileReport(BaseModel):
    profile: MemoryProfile
    verdict: str
    headline: str
    recommendations: list[Recommendation] = Field(default_factory=list)
