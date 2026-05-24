"""MemoryProfiler — agentic GPU memory profiling + optimization advice."""
from .advisor import advise, render_text
from .models import MemoryProfile, ProfileReport, Recommendation

__all__ = [
    "MemoryProfile",
    "Recommendation",
    "ProfileReport",
    "advise",
    "render_text",
]
__version__ = "0.1.0"
