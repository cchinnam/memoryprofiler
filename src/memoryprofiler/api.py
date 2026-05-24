"""FastAPI service for MemoryProfiler.

Run locally (from the repo root, so .env is found):
    pip install -e ".[serve]"          # add ".[llm]" for the Nemotron advisor
    uvicorn memoryprofiler.api:app --reload
    # or: python -m memoryprofiler.api

Endpoints:
    GET  /health   -> liveness
    POST /advise   -> ProfileReport   (no GPU; set "llm": true to use Nemotron)
    POST /profile  -> MemoryProfile   (GPU only — hosted on Modal in Stage 5)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .advisor import advise
from .cli import _load_dotenv
from .models import MemoryProfile, ProfileReport

_load_dotenv()  # pick up NVIDIA_API_KEY for the LLM advisor

app = FastAPI(
    title="MemoryProfiler",
    version="0.1.0",
    description="Profile LLM GPU memory and get optimization recommendations.",
)


class AdviseRequest(BaseModel):
    profile: MemoryProfile
    measured_int8: Optional[MemoryProfile] = None
    llm: bool = False


class ProfileRequest(BaseModel):
    model: str
    max_new_tokens: int = 500
    load_in_8bit: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "memoryprofiler"}


@app.post("/advise", response_model=ProfileReport)
def advise_endpoint(req: AdviseRequest) -> ProfileReport:
    """Turn a memory profile into ranked recommendations. No GPU required."""
    if req.llm:
        from .llm_advisor import advise_llm  # lazy: may need openai + key

        return advise_llm(req.profile, req.measured_int8)
    return advise(req.profile, req.measured_int8)


@app.post("/profile", response_model=MemoryProfile)
def profile_endpoint(req: ProfileRequest) -> MemoryProfile:
    """Profile a model on the GPU this service runs on (needs CUDA + torch)."""
    from .profiler import profile_model  # lazy: torch imported inside

    try:
        return profile_model(
            req.model,
            max_new_tokens=req.max_new_tokens,
            load_in_8bit=req.load_in_8bit,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="GPU dependencies not installed. Install with: pip install -e '.[gpu]'",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"profiling failed (need a CUDA GPU?): {exc}"
        ) from exc


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
