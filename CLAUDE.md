# CLAUDE.md — HF Smol Course

## Project

**HF Smol Course** — learning project following the [Hugging Face Smol Fine-Tuning Language Models course](https://huggingface.co/learn/smol-course/unit0/1). Working through the units hands-on, with the explicit goal of ending the course with a **published, preference-aligned small language model on the HF Hub** that's portfolio-worthy (linkable from [jonathanavni.com/projects](https://jonathanavni.com/projects)).

Hardware: Apple MacBook Pro M4 Pro, 48GB unified memory. The course's primary model is **SmolLM3-3B** (~6GB fp16) — fits on this Mac for LoRA fine-tuning. SmolLM2-135M is the "small-first" smoke/iteration model for fast loops. Heavier runs (full fine-tunes of 3B+, GRPO at scale) will escape to cloud (Colab Pro / Modal / HF Spaces) — decide per-unit, don't default to cloud.

## Core Principles

- **Simplicity first** — prefer the boring obvious solution. Can this be fewer lines? Are abstractions earning their complexity?
- **Push back when warranted** — if an approach has clear problems, say so directly, propose an alternative, accept override. Sycophancy is a failure mode
- **No over-engineering** — don't add features, abstractions, or error handling beyond what's asked. Don't touch code you weren't asked to touch
- **Verification is the #1 lever** — give every task a way to prove it worked (smoke test, eval number, loss curve, generation sample). This 2-3x's output quality
- **Naive-then-optimize** — implement the obviously-correct version first. Verify correctness on a tiny model/dataset. Then scale. Never skip step 1
- **Compaction-safe artifacts** — write important outputs (configs, metrics, decisions) to files immediately. Don't rely on conversation history
- **Learning-first** — this is educational. Prioritize understanding over speed. When a concept is new (LoRA, DPO loss, chat template masking), explain the "why" before implementing

## Workflow

- **Assess before each task** — handle directly, delegate to a subagent, or route to a Hub operation. Consider: (1) does the task benefit from fresh context (complex reasoning, independent scope)? (2) can you work on something else in parallel? (3) is context getting tight? Any of these → delegate. Simple sequential tasks with spacious context → work directly for faster iteration
- When delegating, use **`model: "opus"`** by default. Hold the plan, delegate precise task specs, receive reports back. Workers get fresh context windows
- **Always delegate research and reviews** — research agents explore docs/papers, reviewer agents QA from fresh context (see `/review`). These benefit from isolation regardless of context pressure
- Be precise about specs — "implement SFT training script with `SFTTrainer`, SmolLM3-3B base, `HuggingFaceTB/smoltalk2` dataset, LoRA r=16 on attn + mlp, 3 epochs, seed 42" not "build the SFT trainer"
- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- Use `/start` at session start, `/wrapup` at session end
- Use `/review` after completing a unit, or before any action that's public/expensive (HF Hub push, long cloud run)

## Session Management

- `/clear` between unrelated units; `/compact` to keep focus while clearing noise
- **Two-correction rule**: if wrong twice on the same thing, `/clear` and write a sharper prompt
- Feed raw data (training logs, exception tracebacks, eval metrics) instead of your interpretation
- Use neutral prompts — "search through this training script, follow the data flow, report findings" not "find the bug"

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.12+ |
| Core framework | Hugging Face Transformers + Datasets |
| Fine-tuning | TRL (SFT / DPO / GRPO trainers) |
| Parameter-efficient FT | PEFT (LoRA, QLoRA where applicable) |
| Training loop / devices | Accelerate |
| Experiment tracking | Trackio (HF-native, no account) — WandB optional via extras |
| Hub access | `huggingface_hub` (`hf` CLI) |
| Hardware backend | PyTorch + MPS (Apple Silicon); CUDA when on cloud |
| Testing | pytest (lightweight — this is a learning project) |
| Linting | ruff |
| Package manager | uv |

## Hardware & Compute

- **Local (M4 Pro, 48GB):** SmolLM2-135M for fast smoke iteration; SmolLM3-3B with LoRA for course exercises; DPO on small preference datasets. `torch.device("mps")` works for most TRL trainers; watch for ops that fall back to CPU
- **Known Mac limitations:** `bitsandbytes` (4-bit / 8-bit) is CUDA-only — keep it in the `quant` optional extra, never in base deps. `flash-attn` similarly CUDA-only
- **Cloud escape hatch:** use when (a) full fine-tune of SmolLM3-3B or larger, (b) GRPO with rollouts at scale, (c) need reproducibility on a standard config. Default target: Colab Pro A100 or Modal H100
- **Always state expected runtime + cost before kicking off a run >30 min**

## Key Design Decisions

- **Unit-driven milestones** — each course unit is a milestone in PLAN.md. Don't fork off the curriculum until we've hit the core learning for that unit
- **Course repo is reference, not code** — clone to `course-materials/` (gitignored) for reading notebooks/solutions; write our own implementations in this repo
- **Portfolio aim shapes choices from unit 1** — pick a base model + domain early and keep compounding on it (e.g., SmolLM2-360M → SFT on a specific domain → DPO on human/synthetic preferences → publish). Avoid disconnected one-off notebooks
- **LoRA-first for local runs** — full fine-tunes only when the unit explicitly requires it or we've moved to cloud. Smaller memory footprint, faster iteration, easier to version the adapter
- **Publish-quality bar from day one** — code, commit history, and model cards are part of the portfolio. Don't write "quick hack" code expecting to clean it up later

## Project State Files

- **`PLAN.md`** — single source of truth for active work. Current unit broken out with full detail (goals, tasks, verification checklist). Completed units collapsed to a short summary. Decisions Log is cumulative (never archived). Read at session start, update at session end. **Gitignored** — this is internal, not part of the public repo
- **`PLAN-archive.md`** — full detail of completed units, preserved for historical context. Gitignored
- **`ORIENT.md`** — human onboarding doc. How the project works, common commands. Gitignored
- **`.claude/memory/MEMORY.md`** — accumulated decisions and gotchas as topic-based files. Append after non-obvious choices. Gitignored

## Key Documents

| Document | When to Read |
|----------|-------------|
| `PLAN.md` | Every session start. Current state, active unit, decisions |
| `ORIENT.md` | First-time setup, common commands |
| `.claude/memory/MEMORY.md` | When making a decision touching an area with prior gotchas |

## Course Links

- **Course home:** https://huggingface.co/learn/smol-course/unit0/1
- **Course repo:** https://github.com/huggingface/smol-course (cloned to `course-materials/` via `make clone-course`)
