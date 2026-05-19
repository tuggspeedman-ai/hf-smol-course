---
description: Session close — update project state and capture decisions
---

# /wrapup

End every productive session by persisting what happened.

## Steps

1. **Update `PLAN.md` "Current State" section:**
   - What was accomplished this session
   - What's blocked or needs attention
   - What to do next session
   - Check off completed unit/milestone tasks
   - **If a unit was completed:** move its full detail to `PLAN-archive.md` and collapse it to a short summary in PLAN.md (keep Decisions Log entries in PLAN.md — they're cumulative)

2. **Append to the IN-REPO `.claude/memory/MEMORY.md`** (only if there's something worth remembering):
   - IMPORTANT: This is the file at `<project-root>/.claude/memory/MEMORY.md`, NOT the auto-memory system at `~/.claude/projects/.../memory/MEMORY.md`. The in-repo file is the canonical project memory
   - Non-obvious decisions made and why (e.g., "LoRA r=16 gave materially better eval than r=8 on SmolLM2-360M")
   - Gotchas discovered (e.g., "bitsandbytes import fails on MPS — don't add it to base deps")
   - Conventions established (e.g., "all run names include seed + date")
   - Don't add obvious things. Don't duplicate what's in PLAN.md

3. **Check MEMORY.md size** — if approaching 200 lines, suggest: "MEMORY.md is getting long. Consider running a consolidation pass to move old decisions to a separate file."

4. **Brief summary** — 3-5 sentences of what happened and what's next

## Rules

- Don't skip step 2 even if the session was short — small decisions compound
- Write decisions with enough context for future-you to understand *why* (small models + noisy eval means rationale matters)
- This should take under 2 minutes. If it's taking longer, you're over-documenting
