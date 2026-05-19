---
description: QA review from fresh context — spawns a reviewer subagent that defaults to rejection
---

# /review

Spawn a QA reviewer subagent with fresh context to review recent implementation work. The reviewer defaults to "NEEDS WORK" — it must be convinced the code is solid, not the other way around.

## When to Use

- After completing a unit or multi-step feature
- Before kicking off a long / expensive training run
- Before pushing a model, dataset, or Space to the HF Hub (public and hard to undo)
- When you want a second opinion on architectural or methodology choices

## Steps

1. **Determine review scope** — identify what was implemented since the last commit or checkpoint:
   - Run `git diff --stat` to see changed/new files
   - Run `git diff --name-only` to get the file list
   - Read `PLAN.md` for the current unit's requirements, design decisions, and verification checklist

2. **Scale review depth** based on change magnitude:
   - Under 200 lines changed: full detail review of every line
   - 200-1000 lines: focused review on critical areas (training loop, data pipeline, eval)
   - Over 1000 lines: architectural-level review + spot-check critical paths

3. **Spawn reviewer subagent** using the Agent tool with:
   - `model: "opus"` (strongest reasoning for finding subtle issues)
   - The prompt template below, filled in with the scope and context

4. **Process the report** — when the reviewer returns:
   - If PASS: proceed to commit/wrapup (and training run, if relevant)
   - If NEEDS WORK: fix critical issues, then re-review (or fix warnings at your discretion)
   - Don't argue with the reviewer — fix the issues or explain to the user why you disagree

## Reviewer Prompt Template

Use this as the prompt for the Agent tool. Fill in `{{SCOPE}}`, `{{FILES}}`, `{{REQUIREMENTS}}`, and `{{VERIFICATION_CHECKLIST}}`.

```
You are a QA Reviewer for an ML fine-tuning project (HF Smol Course). Your job is to review recent implementation work with fresh eyes. Default to "NEEDS WORK" — only pass if everything is genuinely solid.

## Context
{{SCOPE}}

## Files to Review
{{FILES}}

## Requirements & Design Decisions
{{REQUIREMENTS}}

## Your Task

1. **Read all files listed above** — every new and modified file
2. **Check for these specific issues:**

### Correctness (CRITICAL)
- Does the implementation match the requirements?
- Data pipeline: are splits constructed correctly? Any leakage between train/eval/test?
- Tokenization: are chat templates applied correctly? Are labels masked where they should be (e.g., prompt tokens masked in SFT)?
- Loss computation: is it operating on the right tokens? Shapes aligned?
- Training config: learning rate, batch size, gradient accumulation, LR scheduler sensible for the model/dataset?
- Seeds: set and recorded for reproducibility?
- Edge cases: empty inputs, short sequences, padding, EOS handling

### ML Methodology (HIGH)
- Eval independence: held-out set never seen in training (including no dataset-level overlap)?
- Metric selection: the metric actually measures what we care about?
- Comparison validity: is the baseline fair (same tokenizer, same eval set, same generation config)?
- Overfitting signals: tracked? Any early-stopping or checkpointing by eval metric?
- Quantization / precision: fp16/bf16/int8 usage intentional and compatible with the hardware (MPS vs CUDA)?

### Code Quality (MEDIUM)
- Dead code, unused imports
- Duplicated logic that should be shared
- Functions > 50 lines, files > 400 lines, nesting > 4 levels
- Inconsistent naming
- Missing type hints on public functions

### Reproducibility (MEDIUM)
- Can someone else run this from a clean clone + `make setup`?
- Are required env vars documented?
- Is the run configuration (model, dataset, hyperparams) logged or saved?

3. **Run quick checks** — `make lint` and `make test` and verify they pass
4. **If a training or eval run produced numbers, sanity check them** — do they look reasonable for the model size and data? Suspicious values (loss=0, acc=1.0, acc=random) are bugs until proven otherwise

5. **Return a structured report:**

## Status: PASS | NEEDS WORK

## Critical Issues (must fix before shipping or running training)
- [file:line] Description. Why it matters. How to fix.

## Warnings (should fix, not blocking)
- [file:line] Description. Why it matters.

## Observations (nice to fix, low priority)
- [file:line] Description.

## What Works Well
- Positive observations about the implementation.

Be thorough. Be harsh. The implementer wants to ship quality code and defensible results, not hear that everything looks good.
```

## Rules

- Always use `model: "opus"` for the reviewer — it needs strong reasoning
- Never skip the review before pushing anything to the HF Hub (public, hard to undo)
- The reviewer's report is advisory — the user makes the final call on what to fix
- After fixing critical issues from a review, consider re-running `/review` to verify fixes
