# %% [markdown]
# # Unit 1 — Exercise 2: Dataset Processing for SFT
#
# Bridge between "understanding the chat template" and "actually training": pure
# data prep, no GPU, no model weights. Three parts:
#   1. Explore the SmolTalk2 dataset structure
#   2. Normalize other dataset formats into the unified chat `messages` layout
#   3. Apply the chat template to produce the `text` column the trainer reads
#
# Run cell-by-cell in VSCode or in stages with
# `uv run python notebooks/unit1/exercise2_dataset_processing.py`.

# %%
# --- Cell 1: scope SmolTalk2 BEFORE downloading it ---
# load_dataset_builder pulls only metadata (split names, row counts, sizes) - a few
# KB - not the actual data. This is the dataset equivalent of the tokenizer trick:
# look before you commit to a multi-GB download.
from datasets import load_dataset_builder

builder = load_dataset_builder("HuggingFaceTB/smoltalk2", "SFT")
splits = builder.info.splits  # dict: split_name -> SplitInfo(num_examples, num_bytes, ...)

if not splits:
    print("No split metadata available from the builder - will inspect via load_dataset.")
else:
    rows = []
    for name, info in splits.items():
        mb = (info.num_bytes or 0) / 1e6
        rows.append((name, info.num_examples, mb))

    total_rows = sum(r[1] for r in rows)
    total_mb = sum(r[2] for r in rows)
    print(f"config 'SFT': {len(rows)} splits, {total_rows:,} rows, {total_mb:,.0f} MB total\n")
    print(f"{'split':<58} {'rows':>10} {'MB':>9}")
    print("-" * 80)
    for name, n, mb in sorted(rows, key=lambda r: r[2]):
        print(f"{name:<58} {n:>10,} {mb:>9,.0f}")

# %%
# --- Cell 2: load two small splits, inspect schema + a sample row each ---
# GOTCHA: load_dataset(repo, "SFT", split="X") resolves the WHOLE 66GB config before
# isolating the split - it tried to download everything. The fix: hf_hub_download the
# exact parquet file (each split == one file), then load_dataset("parquet", ...).
# This gives a real Arrow-backed, memory-mapped Dataset - cheap object, rows on access.
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO = "HuggingFaceTB/smoltalk2"
SFT_FILES = {
    "no_think": "SFT/smoltalk_smollm3_everyday_conversations_no_think-00000-of-00001.parquet",
    "think": "SFT/smoltalk_everyday_convs_reasoning_Qwen3_32B_think-00000-of-00001.parquet",
}
paths = {k: hf_hub_download(REPO, v, repo_type="dataset") for k, v in SFT_FILES.items()}

# split="train" here is just the default name datasets gives a standalone-file load -
# unrelated to smoltalk2's 25 splits.
no_think = load_dataset("parquet", data_files=paths["no_think"], split="train")
think = load_dataset("parquet", data_files=paths["think"], split="train")

print(f"no_think split: {no_think.num_rows:,} rows")
print(f"think split:    {think.num_rows:,} rows")
print(f"\nschema (.features):\n{no_think.features}\n")


def show_row(label, row):
    print(f"\n===== {label} =====")
    print(f"source: {row.get('source')!r}")
    print(f"chat_template_kwargs: {row.get('chat_template_kwargs')!r}")
    print(f"messages ({len(row['messages'])} turns):")
    for i, msg in enumerate(row["messages"]):
        content = msg["content"]
        preview = content if len(content) <= 300 else content[:300] + " ...[truncated]"
        print(f"  [{i}] {msg['role']}: {preview!r}")


show_row("no_think — row 0", no_think[0])
show_row("think — row 0", think[0])

# %%
# --- Cell 3: normalize a non-chat dataset (GSM8K) into the messages layout ---
# GSM8K ships flat question/answer columns - not chat format. To use it for SFT we
# reshape into `messages`. First scope it (habit: builder before load_dataset).
gsm_builder = load_dataset_builder("openai/gsm8k", "main")
for name, info in gsm_builder.info.splits.items():
    print(f"  gsm8k/{name}: {info.num_examples:,} rows, {(info.num_bytes or 0) / 1e6:.1f} MB")

# Small + safe: load a 100-row slice of train.
gsm8k = load_dataset("openai/gsm8k", "main", split="train[:100]")
print(f"\nraw GSM8K columns: {gsm8k.column_names}")
print(f"raw row 0:\n  question: {gsm8k[0]['question']!r}")
print(f"  answer:   {gsm8k[0]['answer']!r}")


# Batched map: `examples` is a dict of column -> list. Return the same shape.
def process_gsm8k(examples):
    processed = []
    for question, answer in zip(examples["question"], examples["answer"]):
        processed.append([
            {"role": "system", "content": "You are a math tutor. Solve problems step by step."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
    return {"messages": processed}


gsm8k_chat = gsm8k.map(process_gsm8k, batched=True, remove_columns=gsm8k.column_names)
print(f"\nafter normalize: columns = {gsm8k_chat.column_names}")
show_row("gsm8k_chat — row 0", gsm8k_chat[0])

# %%
# --- Cell 4: apply the chat template -> the `text` column SFTTrainer reads ---
# add_generation_prompt=False: this is TRAINING data, we want the full conversation
# incl. the assistant's answer. And we pass each row's chat_template_kwargs through,
# so a /no_think row is formatted in /no_think mode (the course omits this).
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")


def to_text(example):
    kwargs = example.get("chat_template_kwargs") or {}
    text = tok.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,  # enable_thinking, custom_instructions, *_tools - empty/default if absent
    )
    return {"text": text}


no_think_fmt = no_think.map(to_text)
gsm8k_fmt = gsm8k_chat.map(to_text)
print(f"no_think_fmt columns: {no_think_fmt.column_names}")
print(f"gsm8k_fmt columns:    {gsm8k_fmt.column_names}")

print("\n===== no_think row 0 -> text (kwargs passed through, enable_thinking=False) =====")
print(no_think_fmt[0]["text"] + "<<END>>")

# The course's way vs the correct way, on the SAME /no_think row.
print("\n===== course's way: no kwargs -> enable_thinking defaults to True =====")
course_way = tok.apply_chat_template(
    no_think[0]["messages"], tokenize=False, add_generation_prompt=False
)
# Show just the system header line that reveals the mismatch.
print("system line:", course_way.splitlines()[5])
print("correct way:", no_think_fmt[0]["text"].splitlines()[5])

print("\n===== gsm8k row 0 -> text (no chat_template_kwargs -> template defaults) =====")
print(gsm8k_fmt[0]["text"][:600] + " ...[truncated]<<END>>")
