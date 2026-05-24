# MemoryProfiler

An agentic tool that profiles **LLM GPU memory** on real hardware and produces
**ranked, quantified optimization recommendations** — quantization, batching,
KV-cache sizing — so teams can cut inference cost without an ML-perf engineer.

## Why

LLM inference *decode* is memory-bandwidth-bound: every output token reads the
entire model from HBM. So GPU memory — not FLOPs — is usually the constraint and
the cost driver. MemoryProfiler measures where the memory actually goes (weights
vs KV cache vs activations), detects whether a workload is memory- or
compute-bound, and recommends the highest-leverage fix.

## Architecture

Two halves with opposite needs:

| Component | Runs on | Needs a GPU? |
|-----------|---------|--------------|
| **Profiler** (`profiler.py`) | Colab / Modal / a GPU box | **Yes** |
| **Advisor** (`advisor.py`) + **CLI** + schemas | anywhere (laptop, CI) | No |

The profiler measures with `torch.cuda` + `nvidia-smi`. The advisor reads the
resulting JSON and emits recommendations — so you can develop and test the brains
of the tool with no GPU at all.

## Quickstart (no GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the advisor on real measured profiles (Phi-3-mini on a Tesla T4):
memoryprofiler advise examples/phi3_t4_fp16.json --int8 examples/phi3_t4_int8.json
```

## Profiling on a GPU

```bash
pip install -e ".[gpu]"            # torch, transformers, accelerate, bitsandbytes
memoryprofiler profile microsoft/Phi-3-mini-4k-instruct            # FP16
memoryprofiler profile microsoft/Phi-3-mini-4k-instruct --int8     # INT8
```

## Measured example (Tesla T4, free Colab)

| | Weights | Peak | Bottleneck |
|---|---|---|---|
| FP16 | 7.64 GB | 7.86 GB | memory-bound |
| INT8 | 4.02 GB | 4.25 GB | compute-bound |
| | **−47%** | **−3.6 GB freed** | *quantization shifted the bottleneck* |

KV cache scales ~linearly with context (~0.41 MB/token measured on Phi-3-mini).

## Stack

Python · Pydantic · PyTorch · Transformers · bitsandbytes ·
NVIDIA NeMo Agent Toolkit + NIM (Nemotron advisor, planned) ·
FastAPI · Modal · Phoenix + OpenTelemetry (planned)

## Roadmap

- [x] Profiler (weights / KV cache / activations, memory-bound detection)
- [x] Rule-based advisor + CLI
- [ ] LLM advisor (Nemotron via NIM) behind the same contract
- [ ] FastAPI service + Phoenix tracing
- [ ] Deploy on Modal (on-demand GPU) / Docker
