# %% [markdown]
# # Unit 1 — Exercise 1: Exploring SmolLM3's Chat Templates
#
# Goal: understand how SmolLM3 turns a list of `{"role", "content"}` messages into
# the exact marker-token string it was trained on, and how the base vs instruct
# variants differ at generation time.
#
# Run cell-by-cell in VSCode (the `# %%` markers render as cells) or in stages with
# `uv run python notebooks/unit1/exercise1_chat_templates.py`.

# %%
# --- Cell 1: setup + device detection + load tokenizers ---
# The chat template lives in the *tokenizer*, not the model weights. So the entire
# template-formatting section costs ~30MB of downloads, not the ~12GB of 3B weights.
import torch
from transformers import AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple MPS")
else:
    device = "cpu"
    print("Using CPU - training will not be practical here")

base_model_name = "HuggingFaceTB/SmolLM3-3B-Base"
instruct_model_name = "HuggingFaceTB/SmolLM3-3B"

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)

print(f"\nbase tokenizer:     {base_model_name}")
print(f"instruct tokenizer: {instruct_model_name}")
print(f"base has chat_template:     {base_tokenizer.chat_template is not None}")
print(f"instruct has chat_template: {instruct_tokenizer.chat_template is not None}")

# %%
# --- Cell 2: special tokens + the raw Jinja template ---
# ChatML wraps each turn in marker tokens. These are *added tokens* in the
# vocabulary - single token IDs, not multi-char strings the model has to spell out.
print("=== instruct tokenizer special tokens ===")
for name in ("bos_token", "eos_token", "pad_token"):
    tok = getattr(instruct_tokenizer, name)
    tid = getattr(instruct_tokenizer, f"{name}_id")
    print(f"  {name:10s} = {tok!r:20s} (id {tid})")

for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
    ids = instruct_tokenizer.encode(marker, add_special_tokens=False)
    print(f"  {marker:14s} -> {ids}  ({'single token' if len(ids) == 1 else 'SPLIT into pieces'})")

print(f"\n=== raw chat_template (Jinja source), {len(instruct_tokenizer.chat_template)} chars ===")
print(instruct_tokenizer.chat_template[:900])
print("  ...[truncated]...")

# Prove the base tokenizer cannot format a conversation - no template to run.
print("\n=== base tokenizer: apply_chat_template ===")
try:
    base_tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    print("  (unexpectedly succeeded)")
except Exception as e:
    print(f"  raises {type(e).__name__}: {e}")

# %%
# --- Cell 3: the rest of the Jinja template ---
# The reasoning-mode logic is the conceptually interesting part. Read it before
# watching it fire on real conversations in cell 4.
print("=== chat_template, chars 900:end ===")
print(instruct_tokenizer.chat_template[900:])

# %%
# --- Cell 4: apply the template to four conversation types ---
# Verify empirically: (1) system block always present, (2) default mode is /think,
# (3) add_generation_prompt appends the assistant-turn opener.
conversations = {
    "simple_qa": [
        {"role": "user", "content": "What is machine learning?"},
    ],
    "with_system": [
        {"role": "system", "content": "You are a helpful AI assistant specialized in explaining technical concepts clearly."},
        {"role": "user", "content": "What is machine learning?"},
    ],
    "multi_turn": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is calculus?"},
        {"role": "assistant", "content": "Calculus is a branch of mathematics that deals with rates of change."},
        {"role": "user", "content": "Can you give me a simple example?"},
    ],
    "reasoning_task": [
        {"role": "user", "content": "Solve step by step: a train travels 120 miles in 2 hours. Average speed?"},
    ],
}


def show(label, text):
    """Print with a visible end-marker so trailing whitespace/newlines are obvious."""
    print(f"\n----- {label} -----")
    print(text + "<<END>>")


for name, messages in conversations.items():
    formatted = instruct_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    show(f"{name}  (add_generation_prompt=True)", formatted)

# For one conversation, show the False vs True delta directly.
print("\n\n========== add_generation_prompt: False vs True (multi_turn) ==========")
for agp in (False, True):
    show(
        f"add_generation_prompt={agp}",
        instruct_tokenizer.apply_chat_template(
            conversations["multi_turn"], tokenize=False, add_generation_prompt=agp
        ),
    )

# %%
# --- Cell 5: flip the reasoning mode two ways ---
msgs = [{"role": "user", "content": "What is 2+2?"}]

show(
    "default (enable_thinking unset)",
    instruct_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
)
show(
    "enable_thinking=False kwarg",
    instruct_tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ),
)
show(
    "/no_think in system message",
    instruct_tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant. /no_think"},
         {"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    ),
)

# %%
# --- Cell 6: the assistant-token mask (chat-template masking) ---
# The {% generation %}...{% endgeneration %} tags in the template let the tokenizer
# emit a per-token mask: 1 = assistant-generated, 0 = prompt/context. SFT computes
# loss ONLY on the mask==1 tokens, so the model learns to *produce* answers, not to
# *parrot* the prompt. This is what "completion-only" / "chat template masking" means.
masked = instruct_tokenizer.apply_chat_template(
    conversations["multi_turn"],
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)
ids = masked["input_ids"]
mask = masked["assistant_masks"]
print(f"total tokens: {len(ids)}   assistant tokens (loss applied): {sum(mask)}   "
      f"context tokens (ignored): {len(mask) - sum(mask)}")

# Group consecutive tokens by mask value so the structure is visible.
print("\n[mask] decoded span")
print("-" * 60)
start = 0
for i in range(1, len(ids) + 1):
    if i == len(ids) or mask[i] != mask[start]:
        span = instruct_tokenizer.decode(ids[start:i])
        flag = "LOSS " if mask[start] else "  -  "
        preview = span.replace("\n", "\\n")
        if len(preview) > 70:
            preview = preview[:67] + "..."
        print(f"[{flag}] {preview}")
        start = i
