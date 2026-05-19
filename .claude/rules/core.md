# Core Rules

## Code Quality

- Many small files (200-400 lines, 800 max). Functions under 50 lines. Max 4 levels of nesting
- Schema-based input validation at system boundaries (CLI args, loaded datasets, HF model configs). Trust internal code
- After refactoring, identify dead code explicitly. Ask before deleting

## Implementation Behavior

- Surface assumptions as a numbered list before non-trivial tasks. "Correct me now or I'll proceed with these"
- When confused, STOP and ask. Name the confusion, present the tradeoff, wait
- Summarize changes after modifications: what changed, what was left alone, any concerns
- Use corrective framing: "you should be doing X — are you still doing it?" beats "remember to do X"

## Safety

- Never hardcode credentials. Reference from .env files (HF_TOKEN, WANDB_API_KEY, etc.)
- Before destructive operations (deleting checkpoints, force-push, `hf repo delete`), confirm with the user
- External communications (git push, `hf upload`, model card publishing) require explicit approval — a pushed model on the Hub is visible immediately

## ML-Specific

- **Set seeds** for any training run intended to be compared or reproduced. Record the seed in the run name or config
- **Guard against data leakage** — splits must be constructed before training, not after. No peeking at eval in training
- **Small-first** — iterate on the tiniest model/dataset that can surface the bug or validate the pattern. Scale up only after the small loop works
- **Cost awareness** — long training runs (>30 min) or cloud GPU use should be approved first. State the expected runtime and compute cost before kicking one off

## Context Hygiene

- After compaction, re-read PLAN.md and relevant files before continuing
- Write important outputs (configs, metrics, decisions) to files immediately
- When switching between unrelated units, suggest `/clear`
- Keep fewer than 10 MCP servers enabled
