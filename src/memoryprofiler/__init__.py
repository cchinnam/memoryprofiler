"""MemoryProfiler — agentic GPU memory profiling + optimization advice."""
from .advisor import advise, render_text
from .llm_advisor import advise_llm
from .models import MemoryProfile, ProfileReport, Recommendation

__all__ = [
    "MemoryProfile",
    "Recommendation",
    "ProfileReport",
    "advise",
    "advise_llm",
    "render_text",
]
__version__ = "0.1.0"
