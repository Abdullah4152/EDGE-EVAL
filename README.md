# EdgeEval: Energy-Aware Evaluation Framework for Fine-tuned LLMs at the Edge

[![ACL Industry Track](https://img.shields.io/badge/ACL-Industry%20Track-blue)](https://aclanthology.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

EdgeEval is a reproducible evaluation framework for measuring the **energy efficiency, economic viability, and task performance** of fine-tuned large language models running on consumer-grade edge hardware (dual Tesla T4 GPUs). It introduces five novel metrics to go beyond accuracy and capture real-world deployment costs.

---

## Table of Contents

1. [What This Framework Does](#what-this-framework-does)
2. [The Five Metrics](#the-five-metrics)
3. [Experiment Setup](#experiment-setup)
4. [Repository Structure](#repository-structure)
5. [Quick Start](#quick-start)
6. [Step-by-Step Guide](#step-by-step-guide)
   - [Step 1: Fine-tuning (LoRA / QLoRA)](#step-1-fine-tuning-lora--qlora)
   - [Step 2: Adapter Merging and Quantization](#step-2-adapter-merging-and-quantization)
   - [Step 3: vLLM Inference and Benchmarking](#step-3-vllm-inference-and-benchmarking)
   - [Step 4: Computing EdgeEval Metrics](#step-4-computing-edgeval-metrics)
7. [Data](#data)
8. [Configuration Reference](#configuration-reference)
9. [Understanding the Results](#understanding-the-results)
10. [Reproducing Paper Results](#reproducing-paper-results)
11. [Citation](#citation)

---

## What This Framework Does

Most LLM benchmarks only measure accuracy. EdgeEval asks a harder question: **Is this model actually worth running locally?**

Given a target hardware environment (dual NVIDIA Tesla T4 GPUs, ~32 GB combined VRAM), EdgeEval evaluates 72 model variants across:

- **2 model families**: LLaMA 3.x and Qwen 2.5
- **3 model sizes**: 1B, 3B, 7–8B parameters
- **2 fine-tuning methods**: LoRA (FP16) and QLoRA (INT4)
- **3 quantization levels at inference**: FP16, INT8, INT4 (AWQ)
- **3 downstream tasks**: RAG (question answering), Summarization, Conversational Chat

The pipeline covers the full lifecycle: fine-tuning → adapter merging → quantization → inference benchmarking → metric computation.

---

## The Five Metrics

### 1. Quantization Fidelity — `Q_ret`
**"How much task accuracy survives quantization?"**

```
Q_ret = (Score_INT4 / Score_FP16) × 100%
```

Scores above 99.9% indicate lossless compression. This tells you whether quantizing a model for deployment is actually safe for your specific task.

### 2. Economic Break-Even — `N_break`
**"How many requests before self-hosting pays off vs. using GPT-4o API?"**

```
N_break = C_train / (C_api_per_req - C_local_per_req)
```

Where costs are computed from measured energy consumption and real electricity/API pricing. A model with `N_break < 20` achieves ROI within the first hour of production traffic.

### 3. Intelligence Per Watt — `IPW`
**"What task score do you get per joule of energy?"**

```
IPW = Normalized_Task_Score (0–1) / Energy_per_Request (J)
```

This is the primary efficiency metric. Higher is better. It lets you compare models of different sizes on a single energy-normalized axis.

### 4. System Density — `ρ_sys`
**"How efficiently does the model use its memory footprint?"**

```
ρ_sys = Throughput (tokens/s) / Model_Size (GB)
```

Small quantized models often achieve dramatically higher `ρ_sys` than large FP16 models, making them preferable for latency-sensitive edge deployments.

### 5. Cold-Start Tax — `C_tax`
**"How many inference requests does one model load cost?"**

```
C_tax = E_load / E_infer
```

Models with `C_tax > 100` are prohibitively expensive for serverless/ephemeral deployments—every cold start costs more energy than 100 real inference calls.

---

## Experiment Setup

| Component | Detail |
|-----------|--------|
| Hardware | 2× NVIDIA Tesla T4 (16 GB each), 32 GB total VRAM |
| Platform | Kaggle (dual-T4 GPU notebooks) |
| Fine-tuning | HuggingFace Transformers + PEFT + TRL |
| Inference Engine | vLLM 0.6.3.post1 |
| Energy Monitoring | pynvml (100ms polling interval) |
| Model Hub | HuggingFace Hub (merged models uploaded post-training) |

**Models fine-tuned (LoRA/QLoRA, 1 epoch each):**

| Family | Sizes |
|--------|-------|
| Qwen 2.5 Instruct | 1.5B, 3B, 7B |
| LLaMA 3.x Instruct | 1B (3.2), 3B (3.2), 8B (3.1) |

**Training data sizes varied by model size** to keep compute roughly equalized:

| Size | RAG (SQuAD) | Summarization (XSum) | Chat (UltraChat) |
|------|-------------|---------------------|------------------|
| 1B | 15,000 rows | 5,000 rows | 10,000 rows |
| 3B | 10,000 rows | 5,000 rows | 10,000 rows |
| 7–8B | 5,000 rows | 5,000 rows | 10,000 rows |

---

## Repository Structure

```
edge-eval/
│
├── README.md                        ← You are here
├── requirements.txt                 ← All Python dependencies
├── .env.example                     ← Template for environment variables (HF tokens)
│
├── src/
│   ├── train/
│   │   ├── train_single_gpu.py      ← Single GPU LoRA/QLoRA fine-tuning loop
│   │   ├── train_dual_gpu.py        ← Dual GPU (device_map=auto) fine-tuning loop
│   │   ├── data_loader.py           ← Load & format SQuAD, XSum, UltraChat
│   │   └── energy_monitor.py        ← pynvml-based GPU energy tracking (multi-GPU)
│   │
│   ├── merge/
│   │   ├── merge_and_quantize.py    ← Merge LoRA adapters + run INT8/INT4 quantization
│   │   └── upload_to_hub.py         ← Upload merged models to HuggingFace Hub
│   │
│   ├── inference/
│   │   ├── run_inference.py         ← vLLM batch inference for all 72 model variants
│   │   ├── generate_model_list.py   ← Generate the full 72-model config list
│   │   └── eval_metrics.py          ← ROUGE, F1, BERTScore, LLM-as-judge scoring
│   │
│   ├── metrics/
│   │   ├── compute_all_metrics.py   ← Main script: computes all 5 EdgeEval metrics
│   │   ├── q_ret.py                 ← Quantization Fidelity (Q_ret)
│   │   ├── n_break.py               ← Economic Break-Even (N_break)
│   │   ├── ipw.py                   ← Intelligence Per Watt (IPW)
│   │   ├── rho_sys.py               ← System Density (ρ_sys)
│   │   └── c_tax.py                 ← Cold-Start Tax (C_tax)
│   │
│   └── analysis/
│       └── summary_report.py        ← Generate the combined metrics summary report
│
├── data/
│   ├── eval/
│   │   ├── rag_eval_gold.jsonl      ← RAG evaluation set (40 samples from SQuAD)
│   │   ├── summ_eval_gold.jsonl     ← Summarization evaluation set (40 samples from XSum)
│   │   └── chat_eval_gold.jsonl     ← Chat evaluation set (40 samples from UltraChat)
│   └── README_data.md               ← Notes on dataset sources and splits
│
├── results/
│   ├── vllm_inference_metrics_tesla_t4_dual.csv
│   ├── training_energy_log_llama.csv
│   ├── training_energy_log_qwen.csv
│   ├── merging_quantization_energy_llama.csv
│   ├── merging_quantization_energy_qwen.csv
│   ├── final_36_variants_report_llama.csv
│   ├── final_36_variants_report_qwen.csv
│   ├── q_ret.csv
│   ├── n_break.csv
│   ├── ipw.csv
│   ├── rho_sys.csv
│   ├── c_tax.csv
│   └── metrics_all_task.jsonl
│
└── notebooks/
    └── acl-final-anonymous.ipynb    ← Original end-to-end experiment notebook
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s) with at least 16 GB VRAM (dual T4 recommended)
- HuggingFace account and API token(s)
- Access to `meta-llama/Llama-3.x` models (requires HF approval)

### Install

```bash
git clone https://github.com/your-username/edge-eval.git
cd edge-eval
pip install -r requirements.txt
```

### Set up your HuggingFace tokens

```bash
cp .env.example .env
# Edit .env with your HuggingFace tokens
```

Your `.env` file should look like:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_USERNAME=your_hf_username
```

If you are using separate accounts for Llama and Qwen models (as in the original paper setup), you can set:
```
QWEN_HF_TOKEN=hf_...
QWEN_HF_USERNAME=...
LLAMA_HF_TOKEN=hf_...
LLAMA_HF_USERNAME=...
```

### Run the full pipeline

```bash
# Step 1: Fine-tune all models (takes hours — run on GPU)
python src/train/train_dual_gpu.py --config config_dual_t4.yaml

# Step 2: Merge adapters and quantize
python src/merge/merge_and_quantize.py

# Step 3: Run vLLM inference benchmark
python src/inference/run_inference.py

# Step 4: Compute all 5 EdgeEval metrics
python src/metrics/compute_all_metrics.py

# Step 5: Generate summary report
python src/analysis/summary_report.py
```

---

## Step-by-Step Guide

### Step 1: Fine-tuning (LoRA / QLoRA)

The fine-tuning script trains each model with both LoRA (FP16) and QLoRA (INT4) on all three tasks. For 7B/8B models, it uses `device_map="auto"` to distribute the model across both GPUs.

```bash
python src/train/train_dual_gpu.py
```

**What this script does:**
- Loads the base model (from HuggingFace Hub) with optional 4-bit quantization (QLoRA)
- Applies LoRA adapters to all attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- Formats training data using the model's native chat template
- Trains for 1 epoch, logging energy usage per GPU at 500ms intervals
- Saves the adapter weights locally

**Key LoRA configuration:**
```python
LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
```

**Dual GPU note:** For 7B/8B models, memory is split across both GPUs (13 GB each) using `device_map="auto"`. Smaller 1B and 3B models fit on a single GPU.

**Energy output:** Saved to `training_energy_log.csv` with per-GPU Joules, average Watts, total seconds, and number of training rows.

---

### Step 2: Adapter Merging and Quantization

After training, LoRA adapters are merged into the base model weights and then quantized for efficient inference.

```bash
python src/merge/merge_and_quantize.py
```

**What this produces for each trained LoRA adapter:**
- `Model-FP16`: Merged model in full FP16 precision
- `Model-INT8`: INT8 quantized version (using bitsandbytes)
- `Model-INT4`: INT4 quantized version (using AWQ)

**For QLoRA adapters:**
- `Model-MERGED`: The QLoRA model merged back to FP16 (one variant only)

All merged models are uploaded directly to HuggingFace Hub to avoid local disk constraints. Energy consumption during merging and quantization is also tracked and logged to `merging_quantization_energy.csv`.

---

### Step 3: vLLM Inference and Benchmarking

All 72 merged model variants are benchmarked using vLLM for production-representative throughput and latency measurements.

```bash
python src/inference/run_inference.py
```

**What is measured per model:**
- Model load time (seconds)
- Time to First Token (TTFT) — mean and median across 40 samples
- Inter-Token Latency (ITL) — mean and median
- Overall throughput (tokens/second)
- Total energy consumed (Joules) via pynvml
- Average and peak power draw (Watts)

**Evaluation data:** Each task uses 40 held-out gold samples from the same source datasets used in training (SQuAD for RAG, XSum for summarization, UltraChat for chat), saved in `data/eval/`.

**Task scoring:**
- **RAG**: NLI Entailment (Context → Generation) (Honovich et al., 2022) and F1 score between model answer and gold answer span
- **Summarization**: NLI Non-Contradiction and ROUGE-L  (Lin, 2004); against the reference summary
- **Chat**: LLM-as-a-Judge (GPT-4o) (Zheng et al., 2023) ratings (Zheng et al., 2023) on Helpfulness and Safety (1–10 Likert scale)

Results are saved to `vllm_inference_metrics_tesla_t4_dual.csv`.

> **Note on vLLM version:** Use vLLM 0.6.3.post1 with torch 2.4.0 (CUDA 12.1). See `requirements.txt` for the exact install order. Newer versions may have compatibility issues with dual T4 setups.

---

### Step 4: Computing EdgeEval Metrics

Once inference results and performance scores are collected, run the metrics computation:

```bash
python src/metrics/compute_all_metrics.py \
  --inference_csv results/vllm_inference_metrics_tesla_t4_dual.csv \
  --perf_llama results/final_36_variants_report_llama.csv \
  --perf_qwen results/final_36_variants_report_qwen.csv \
  --train_llama results/training_energy_log_llama.csv \
  --train_qwen results/training_energy_log_qwen.csv \
  --output_dir results/
```

This produces all five metric CSVs (`q_ret.csv`, `n_break.csv`, `ipw.csv`, `rho_sys.csv`, `c_tax.csv`) and a combined summary report.

**Economic constants used in N_break (configurable):**
```python
ELECTRICITY_COST_PER_KWH = 0.12   # USD (US average)
GPT4O_COST_PER_REQUEST   = 0.005  # USD (GPT-4o API, ~500 tokens)
```

---

## Data

### Training Data (Public Datasets, Not Included)

Training data is drawn directly from HuggingFace Hub at runtime:

| Task | Dataset | Split | Rows Used |
|------|---------|-------|-----------|
| RAG | `rajpurkar/squad` | `train` | 5k–15k (by model size) |
| Summarization | `EdinburghNLP/xsum` | `train` | 5,000 |
| Chat | `HuggingFaceH4/ultrachat_200k` | `train_sft` | 10,000 |

Because these are standard public benchmarks already on HuggingFace, we do not redistribute the training splits in this repository. The data loading script (`src/train/data_loader.py`) pulls and formats them automatically.

### Evaluation Data (Included)

The 40-sample held-out evaluation sets for each task are included in `data/eval/`. These are small fixed subsets drawn from the same datasets, formatted as JSONL. Including them ensures exact reproducibility of inference benchmarks.

**Should you upload your full data splits?** Since your training data comes from existing public datasets and you are only taking a subset, you do **not** need to include the full training splits in this repo. Instead, the data loading script re-creates the exact same splits at runtime using fixed random seeds or fixed `[:N]` slicing. The eval JSONL files are small enough (~a few MB) and are the only data files that need to be included.

---

## Configuration Reference

### Hardware Environments

Set `HARDWARE_ENV` in the training script:

```python
HARDWARE_ENV = "T4"       # Single GPU (1B, 3B models)
HARDWARE_ENV = "T4_DUAL"  # Dual GPU (7B/8B models, device_map=auto)
```

### Row Counts by Model Size

```python
ROW_COUNTS = {
    "1B":  {"rag": 15000, "summ": 5000, "chat": 10000},
    "3B":  {"rag": 10000, "summ": 5000, "chat": 10000},
    "7B":  {"rag": 5000,  "summ": 5000, "chat": 10000},
}
```

### vLLM Settings

```python
SAMPLES_PER_TASK   = 40       # Evaluation samples per task
MAX_MODEL_LEN      = 2048     # Max sequence length
GPU_MEMORY_UTIL    = 0.85     # GPU memory fraction for vLLM
TENSOR_PARALLEL    = 2        # Number of GPUs for tensor parallelism
```

---

## Understanding the Results

### Result Files

| File | Description |
|------|-------------|
| `training_energy_log_*.csv` | Per-model training energy (Joules, Watts, seconds) |
| `merging_quantization_energy_*.csv` | Energy cost to merge adapters and quantize |
| `final_36_variants_report_*.csv` | Task performance scores (primary + secondary metrics) |
| `vllm_inference_metrics_tesla_t4_dual.csv` | Full inference benchmark: throughput, latency, energy, power |
| `q_ret.csv` | Quantization Fidelity scores for all 36 quantized variants |
| `n_break.csv` | Economic Break-Even request counts |
| `ipw.csv` | Intelligence Per Watt scores |
| `rho_sys.csv` | System Density (throughput / model size) |
| `c_tax.csv` | Cold-Start Tax ratios and serverless viability flags |

### Key Findings (from the paper)

- INT4 quantized models retain **>99.9% task accuracy** on RAG and Summarization vs FP16 baselines
- 1B INT4 models achieve the highest IPW, making them the most energy-efficient option for simple tasks
- Cold-Start Tax exceeds 100× for all 7B+ models, making them **prohibitive for serverless deployments**
- N_break for 1B INT4 models is often **< 20 requests**, meaning ROI is achieved within the first hour of traffic
- System density (ρ_sys) of 1B INT4 models is approximately 3–4× higher than 7B FP16 models

---

## Reproducing Paper Results

To reproduce the exact paper results without re-running training:

1. **Download the pre-trained merged models** from HuggingFace Hub (links will be added after de-anonymization)
2. **Use the provided evaluation sets** in `data/eval/`
3. **Run only the inference and metrics steps:**

```bash
python src/inference/run_inference.py --use_paper_models
python src/metrics/compute_all_metrics.py
```

The pre-computed result CSVs in `results/` already contain all values reported in the paper tables and can be used for independent verification.

---

## Citation

If you use EdgeEval in your research, please cite:

```bibtex
@inproceedings{edgeeval2025,
  title     = {Are Large Language Models Edge-Ready? An Energy-Aware Evaluation Framework},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Industry Track)},
  year      = {2025}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The datasets used (SQuAD, XSum, UltraChat) are subject to their own respective licenses. Please review them before commercial use.
