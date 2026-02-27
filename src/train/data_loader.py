"""
data_loader.py
--------------
Load and format training data for three tasks:
  - rag:  SQuAD (rajpurkar/squad)
  - summ: XSum  (EdinburghNLP/xsum)
  - chat: UltraChat 200k (HuggingFaceH4/ultrachat_200k)

Data is formatted using each model's native chat template via
tokenizer.apply_chat_template().
"""

from datasets import load_dataset, Dataset

# ─────────────────────────────────────────────────────────
# System prompts (injected as the system turn in every sample)
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "rag": (
        'You are a factual assistant. Use ONLY the provided context to answer the question.\n'
        'If the answer is not contained in the context, respond with '
        '"I do not have enough information."'
    ),
    "summ": (
        "Write exactly one sentence that serves as a news-style lead summarizing "
        "the most important event in the article.\n"
        "Do not use multiple sentences. Do not add information not present in the article."
    ),
    "chat": "You are a helpful and coherent AI assistant.",
}

# ─────────────────────────────────────────────────────────
# Dataset configs
# ─────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "rag":  ("rajpurkar/squad", "train", "main"),
    "summ": ("EdinburghNLP/xsum", "train", "main"),
    "chat": ("HuggingFaceH4/ultrachat_200k", "train_sft", "main"),
}


def load_and_prep_data(task: str, row_count: int, tokenizer) -> Dataset | None:
    """
    Load raw data from HuggingFace Hub and return an unformatted Dataset.
    Formatting is done separately via format_data_using_chat_template().

    Args:
        task:       One of "rag", "summ", "chat".
        row_count:  Number of training rows to pull.
        tokenizer:  The model tokenizer (not used for loading, kept for API symmetry).

    Returns:
        A HuggingFace Dataset, or None on failure.
    """
    print(f"   ...Loading {task.upper()} ({row_count} rows)...")

    ds_id, default_split, rev = DATASET_CONFIGS[task]

    try:
        if task == "rag":
            # SQuAD supports direct slicing — fast
            print(f"      📥 Loading SQuAD dataset...")
            ds = load_dataset(ds_id, split=f"train[:{int(row_count)}]", revision=rev)
            print(f"      ✅ Loaded {len(ds)} rows.")
            return ds

        # XSum and UltraChat are large — use streaming
        try:
            ds_stream = load_dataset(ds_id, split=default_split, streaming=True, revision=rev)
        except ValueError:
            ds_stream = load_dataset(ds_id, split="train", streaming=True, revision=rev)

        data_head = []
        max_iterations = row_count * 2
        print(f"      📥 Streaming {row_count} samples...")

        for iter_count, item in enumerate(ds_stream, start=1):
            data_head.append(item)
            if iter_count % 1000 == 0:
                print(f"      ... {iter_count} collected")
            if len(data_head) >= row_count:
                break
            if iter_count > max_iterations:
                print(f"      ⚠️  Safety limit hit at {iter_count} iterations")
                break

        if not data_head:
            print("      ❌ No data collected.")
            return None

        ds = Dataset.from_list(data_head)
        print(f"      ✅ Loaded {len(ds)} rows.")
        return ds

    except Exception as e:
        import traceback
        print(f"      ❌ LOAD ERROR: {e}")
        traceback.print_exc()
        return None


def format_data_using_chat_template(examples: dict, tokenizer, task: str) -> dict:
    """
    Format a batch of raw examples into chat-template strings for SFT.

    This function is intended to be used with datasets.map(batched=True).
    `examples` is therefore a dict-of-lists (NOT a Dataset object).

    Returns:
        {"text": [str, ...]} — list of formatted strings ready for SFTTrainer.
    """
    texts = []
    system_msg = SYSTEM_PROMPTS[task]

    # ── RAG (SQuAD) ───────────────────────────────────────────────────────
    if task == "rag":
        for ctx, q, ans in zip(
            examples["context"], examples["question"], examples["answers"]
        ):
            try:
                if not ctx or not q or not ans:
                    continue
                # SQuAD answers are stored as {"text": [...], "answer_start": [...]}
                if isinstance(ans, dict) and "text" in ans:
                    answer_text = ans["text"][0] if ans["text"] else None
                else:
                    answer_text = str(ans)
                if not answer_text:
                    continue

                conversation = [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": f"### Context:\n{ctx}\n\n### Question:\n{q}",
                    },
                    {"role": "assistant", "content": answer_text},
                ]
                texts.append(
                    tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=False
                    )
                )
            except Exception:
                continue

    # ── SUMMARIZATION (XSum) ──────────────────────────────────────────────
    elif task == "summ":
        for doc, summ in zip(examples["document"], examples["summary"]):
            if not doc or not summ:
                continue
            conversation = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"### Article:\n{doc}"},
                {"role": "assistant", "content": summ},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )

    # ── CHAT (UltraChat) ──────────────────────────────────────────────────
    elif task == "chat":
        for conversation in examples["messages"]:
            if not conversation:
                continue
            conv_obj = [msg.copy() for msg in conversation if isinstance(msg, dict)]
            if not conv_obj:
                continue
            if conv_obj[0]["role"] != "system":
                conv_obj.insert(0, {"role": "system", "content": system_msg})
            texts.append(
                tokenizer.apply_chat_template(
                    conv_obj, tokenize=False, add_generation_prompt=False
                )
            )

    return {"text": texts}
