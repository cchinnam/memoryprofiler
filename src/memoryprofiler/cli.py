"""Command-line interface for MemoryProfiler.

    memoryprofiler advise examples/phi3_t4_fp16.json --int8 examples/phi3_t4_int8.json
    memoryprofiler profile microsoft/Phi-3-mini-4k-instruct        # needs a GPU
"""
from __future__ import annotations

import argparse
import json

from .advisor import advise, render_text
from .models import MemoryProfile


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): sets vars not already in env."""
    import os

    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="memoryprofiler",
        description="Profile LLM GPU memory and get optimization recommendations.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("advise", help="run the advisor on a profile JSON (no GPU needed)")
    a.add_argument("profile_json")
    a.add_argument(
        "--int8",
        dest="int8_json",
        default=None,
        help="optional measured INT8 profile JSON for exact savings",
    )
    a.add_argument(
        "--llm",
        action="store_true",
        help="use the Nemotron LLM advisor (needs NVIDIA_API_KEY); "
        "falls back to rule-based if unavailable",
    )

    pr = sub.add_parser("profile", help="profile a model on a GPU (needs CUDA + torch)")
    pr.add_argument("model")
    pr.add_argument("--int8", action="store_true", help="load the model in INT8")
    pr.add_argument("--max-new-tokens", type=int, default=500)

    args = parser.parse_args()

    if args.cmd == "advise":
        with open(args.profile_json) as f:
            profile = MemoryProfile(**json.load(f))
        int8 = None
        if args.int8_json:
            with open(args.int8_json) as f:
                int8 = MemoryProfile(**json.load(f))
        if args.llm:
            from .llm_advisor import advise_llm  # lazy: only path that may need openai

            report = advise_llm(profile, int8)
        else:
            report = advise(profile, int8)
        print(render_text(report))

    elif args.cmd == "profile":
        from .profiler import profile_model  # lazy import: needs torch + GPU

        profile = profile_model(
            args.model, load_in_8bit=args.int8, max_new_tokens=args.max_new_tokens
        )
        print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
