# Data

## Evaluation Sets (included)

The `eval/` directory contains the 40-sample held-out evaluation sets used in the paper:

| File | Task | Source Dataset | Samples |
|------|------|----------------|---------|
| `rag_eval_gold.jsonl`  | RAG           | SQuAD (rajpurkar/squad) | 40 |
| `summ_eval_gold.jsonl` | Summarization | XSum (EdinburghNLP/xsum) | 40 |
| `chat_eval_gold.jsonl` | Chat          | UltraChat 200k (HuggingFaceH4/ultrachat_200k) | 40 |

Each line is a JSON object. Schemas:

**rag_eval_gold.jsonl**
```json
{"context": "...", "question": "...", "answers": {"text": ["answer"], "answer_start": [42]}}
```

**summ_eval_gold.jsonl**
```json
{"document": "...", "summary": "One sentence reference summary."}
```

**chat_eval_gold.jsonl**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Training Data (not included)

Training data is streamed from HuggingFace at runtime by `src/train/data_loader.py`:

| Task | Dataset ID | Split | Rows (by model size) |
|------|-----------|-------|----------------------|
| RAG  | `rajpurkar/squad` | train | 5k (7B), 10k (3B), 15k (1B) |
| Summarization | `EdinburghNLP/xsum` | train | 5,000 all sizes |
| Chat | `HuggingFaceH4/ultrachat_200k` | train_sft | 10,000 all sizes |

These datasets are publicly available on HuggingFace under their respective licenses.
Please review each dataset's license before commercial use.
