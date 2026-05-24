"""Deploy MemoryProfiler on Modal.

    pip install modal
    modal setup                                   # one-time browser auth
    modal secret create nvidia NVIDIA_API_KEY=nvapi-...   # for the LLM advisor
    modal deploy modal_app.py                      # -> public URLs

Architecture (cost-smart):
  * The FastAPI web app (/health, /advise) runs CPU-only and scales to zero.
  * /profile dispatches to a separate GPU function (A10G) that spins up only
    for the call and tears down after — pennies per profile, $0 when idle.
"""
import modal

app = modal.App("memoryprofiler")

# CPU image for the web app + advisors (no torch).
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "pydantic>=2", "openai>=1.0")
    .add_local_python_source("memoryprofiler")
)

# GPU image for the profiler (torch + transformers + quantization).
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers", "accelerate", "bitsandbytes", "pydantic>=2"
    )
    .add_local_python_source("memoryprofiler")
)

# NVIDIA key for the LLM advisor:  modal secret create nvidia NVIDIA_API_KEY=...
nvidia_secret = modal.Secret.from_name("nvidia")


@app.function(image=gpu_image, gpu="A10G", timeout=600)
def profile_remote(
    model: str, max_new_tokens: int = 500, load_in_8bit: bool = False
) -> dict:
    """Run the profiler on an on-demand A10G and return the profile as a dict."""
    from memoryprofiler.profiler import profile_model

    return profile_model(
        model, max_new_tokens=max_new_tokens, load_in_8bit=load_in_8bit
    ).model_dump()


@app.function(image=web_image, secrets=[nvidia_secret])
@modal.asgi_app()
def fastapi_app():
    from typing import Optional

    from fastapi import FastAPI
    from pydantic import BaseModel

    from memoryprofiler.advisor import advise
    from memoryprofiler.models import MemoryProfile, ProfileReport

    web = FastAPI(title="MemoryProfiler", version="0.1.0")

    class AdviseRequest(BaseModel):
        profile: MemoryProfile
        measured_int8: Optional[MemoryProfile] = None
        llm: bool = False

    class ProfileRequest(BaseModel):
        model: str
        max_new_tokens: int = 500
        load_in_8bit: bool = False

    @web.get("/health")
    def health():
        return {"status": "ok", "service": "memoryprofiler", "platform": "modal"}

    @web.post("/advise", response_model=ProfileReport)
    def advise_endpoint(req: AdviseRequest):
        if req.llm:
            from memoryprofiler.llm_advisor import advise_llm

            return advise_llm(req.profile, req.measured_int8)
        return advise(req.profile, req.measured_int8)

    @web.post("/profile", response_model=MemoryProfile)
    def profile_endpoint(req: ProfileRequest):
        data = profile_remote.remote(
            req.model, req.max_new_tokens, req.load_in_8bit
        )
        return MemoryProfile(**data)

    return web
