"""
run_inference.py
----------------
vLLM-based inference benchmark for all 72 model variants on dual Tesla T4.

Measures per model:
  - Model load time (cold-start)
  - Time to First Token (TTFT) — mean & median across 40 samples
  - Inter-Token Latency (ITL) — mean & median
  - Overall throughput (tokens/s)
  - Total energy (J), average power (W), peak power (W)

Results are written to:
  vllm_inference_metrics_tesla_t4_dual.csv  — full metrics table
  inference_results/<model>_predictions.jsonl — per-sample predictions

Requirements:
  vllm==0.6.3.post1, torch==2.4.0 (CUDA 12.1)

Usage:
    QWEN_HF_TOKEN=<tok> LLAMA_HF_TOKEN=<tok> HF_USERNAME=<user> \
        python src/inference/run_inference.py
"""

import os
import gc
import csv
import json
import time
import statistics
from typing import Dict, List

import torch

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "vLLM not found. Install with:\n"
        "  pip install numpy==1.26.3\n"
        "  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121\n"
        "  pip install vllm==0.6.3.post1"
    )

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from generate_model_list import generate_all_models, patch_qwen2_tokenizer

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
SAMPLES_PER_TASK  = 40
MAX_MODEL_LEN     = 2048
GPU_MEMORY_UTIL   = 0.70
TENSOR_PARALLEL   = torch.cuda.device_count()

DATASET_BASE = os.getenv("DATASET_BASE", "./data/eval")
DATASETS = {
    "chat": os.path.join(DATASET_BASE, "chat_eval_gold.jsonl"),
    "rag":  os.path.join(DATASET_BASE, "rag_eval_gold.jsonl"),
    "summ": os.path.join(DATASET_BASE, "summ_eval_gold.jsonl"),
}

OUTPUT_DIR  = "./inference_results"
METRICS_LOG = os.path.join(OUTPUT_DIR, "vllm_inference_metrics_tesla_t4_dual.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPTS = {
    "rag":  "You are a factual assistant. Use ONLY the provided context to answer the question.",
    "summ": "Write exactly one sentence that summarizes the most important event.",
    "chat": "You are a helpful AI assistant.",
}

print(f"✅ vLLM ready | GPUs: {TENSOR_PARALLEL} | Tensor parallel: {TENSOR_PARALLEL}")

# ─────────────────────────────────────────────────────────
# Energy monitor
# ─────────────────────────────────────────────────────────
import threading

class EnergyMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval       = interval
        self.running        = False
        self.total_joules   = 0.0
        self.peak_watts     = 0.0
        self.min_watts      = float("inf")
        self.power_samples  = []
        self.duration       = 0.0
        self.available      = HAS_PYNVML

        if self.available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.available = False

    def _monitor(self):
        start = time.time()
        while self.running and self.available:
            try:
                pw = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.power_samples.append(pw)
                self.peak_watts   = max(self.peak_watts, pw)
                self.min_watts    = min(self.min_watts,  pw)
                self.total_joules += pw * self.interval
            except Exception:
                pass
            time.sleep(self.interval)
        self.duration = time.time() - start

    def start(self):
        if self.available:
            self.running = True
            self.thread  = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def stop(self) -> Dict[str, float]:
        self.running = False
        if self.available and hasattr(self, "thread"):
            self.thread.join(timeout=2)
        if not self.available or not self.power_samples:
            return {"total_joules": 0.0, "avg_watts": 0.0, "peak_watts": 0.0, "duration": 0.0}
        avg_watts = self.total_joules / self.duration if self.duration > 0 else 0.0
        return {
            "total_joules": self.total_joules,
            "avg_watts":    avg_watts,
            "peak_watts":   self.peak_watts,
            "duration":     self.duration,
        }

# ─────────────────────────────────────────────────────────
# Data loading & prompt formatting
# ─────────────────────────────────────────────────────────
def load_eval_data(task: str, num_samples: int) -> List[Dict]:
    data = []
    try:
        with open(DATASETS[task], "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data.append(json.loads(line.strip()))
        print(f"   ✅ Loaded {len(data)} {task} samples")
    except Exception as e:
        print(f"   ❌ Failed to load {task}: {e}")
    return data


def format_prompt(item: Dict, task: str) -> str:
    sys = SYSTEM_PROMPTS[task]
    if task == "rag":
        return f"{sys}\n\nContext: {item.get('context','')}\n\nQuestion: {item.get('question','')}\n\nAnswer:"
    elif task == "summ":
        return f"{sys}\n\nArticle: {item.get('document','')}\n\nSummary:"
    elif task == "chat":
        msgs = item.get("messages", [])
        user = msgs[-1].get("content", "") if msgs else item.get("prompt", "")
        return f"{sys}\n\nUser: {user}\n\nAssistant:"
    return ""

# ─────────────────────────────────────────────────────────
# Token metrics
# ─────────────────────────────────────────────────────────
def calculate_token_metrics(output, inf_start: float) -> Dict[str, float]:
    try:
        m         = output.metrics
        ttft      = m.first_token_time - m.first_scheduled_time
        n_toks    = len(output.outputs[0].token_ids)
        total_t   = m.finished_time - m.first_scheduled_time
        itl       = (total_t - ttft) / (n_toks - 1) if n_toks > 1 else 0.0
        return {
            "ttft_ms":       ttft * 1000,
            "itl_ms":        itl * 1000,
            "total_time_s":  total_t,
            "num_tokens":    n_toks,
            "tokens_per_sec": n_toks / total_t if total_t > 0 else 0.0,
        }
    except Exception:
        total_t = time.time() - inf_start
        n_toks  = len(output.outputs[0].token_ids) if output.outputs else 0
        return {
            "ttft_ms": 0.0, "itl_ms": 0.0,
            "total_time_s": total_t, "num_tokens": n_toks,
            "tokens_per_sec": n_toks / total_t if total_t > 0 else 0.0,
        }

# ─────────────────────────────────────────────────────────
# CSV initialisation
# ─────────────────────────────────────────────────────────
def init_logs():
    if not os.path.exists(METRICS_LOG):
        with open(METRICS_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "Model_ID", "Family", "Size", "Method", "Precision", "Task",
                "Quantization", "Num_Samples", "Total_Tokens",
                "Model_Load_Time", "Inference_Time", "Overall_Throughput",
                "TTFT_Mean", "TTFT_Median", "ITL_Mean", "ITL_Median",
                "Energy_Total_J", "Power_Avg_W", "Power_Peak_W", "Status",
            ])

# ─────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────
def run_inference(model_config: Dict) -> Dict:
    model_id    = model_config["model_id"]
    task        = model_config["task"]
    quantization= model_config.get("quantization")

    print(f"\n{'='*70}")
    print(f"🔬 MODEL: {model_id}")
    print(f"   Task: {task} | Precision: {model_config['precision']}")
    print(f"{'='*70}")

    eval_data = load_eval_data(task, SAMPLES_PER_TASK)
    if not eval_data:
        return {"model_id": model_id, "status": "failed", "error": "No data"}

    prompts = [p for item in eval_data if (p := format_prompt(item, task))]
    print(f"   📝 {len(prompts)} prompts ready")

    llm     = None
    outputs = None
    try:
        vllm_kwargs = {
            "model":                  model_id,
            "trust_remote_code":      True,
            "dtype":                  "float16",
            "gpu_memory_utilization": GPU_MEMORY_UTIL,
            "max_model_len":          MAX_MODEL_LEN,
            "tensor_parallel_size":   TENSOR_PARALLEL,
            "skip_tokenizer_init":    True,   # avoids Qwen2 tokenizer bug
        }
        if quantization:
            vllm_kwargs["quantization"] = quantization

        monitor    = EnergyMonitor()
        monitor.start()

        load_start = time.time()
        llm        = LLM(**vllm_kwargs)
        load_time  = time.time() - load_start
        print(f"   ✅ Loaded in {load_time:.2f}s")

        sampling_params = SamplingParams(
            temperature=0.1 if task in ("rag", "summ") else 0.7,
            top_p=0.9,
            max_tokens=128 if task == "summ" else 256,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
        )

        inf_start = time.time()
        outputs   = llm.generate(prompts, sampling_params)
        inf_time  = time.time() - inf_start

        energy = monitor.stop()

        ttft_vals   = []
        itl_vals    = []
        total_tokens= 0
        for out in outputs:
            m = calculate_token_metrics(out, inf_start)
            ttft_vals.append(m["ttft_ms"])
            itl_vals.append(m["itl_ms"])
            total_tokens += m["num_tokens"]

        throughput = total_tokens / inf_time if inf_time > 0 else 0.0
        print(f"   📊 TTFT: {statistics.mean(ttft_vals):.2f}ms | Throughput: {throughput:.2f} tok/s")

        # Save predictions
        pred_file = os.path.join(OUTPUT_DIR, f"{model_id.replace('/', '_')}_predictions.jsonl")
        with open(pred_file, "w") as f:
            for item, out in zip(eval_data, outputs):
                f.write(json.dumps({
                    "model":      model_id,
                    "prediction": out.outputs[0].text.strip(),
                    "gold":       item.get("answer", item.get("summary", "")),
                }, ensure_ascii=False) + "\n")

        # Log metrics
        with open(METRICS_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                model_id, model_config["family"], model_config["size"],
                model_config["method"], model_config["precision"], task,
                quantization or "None", len(prompts), total_tokens,
                f"{load_time:.2f}", f"{inf_time:.2f}", f"{throughput:.2f}",
                f"{statistics.mean(ttft_vals):.2f}", f"{statistics.median(ttft_vals):.2f}",
                f"{statistics.mean(itl_vals):.2f}",  f"{statistics.median(itl_vals):.2f}",
                f"{energy['total_joules']:.2f}", f"{energy['avg_watts']:.2f}",
                f"{energy['peak_watts']:.2f}", "SUCCESS",
            ])

        return {"model_id": model_id, "status": "success"}

    except Exception as e:
        import traceback
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return {"model_id": model_id, "status": "failed", "error": str(e)}
    finally:
        for obj in [llm, outputs]:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(3)

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    patch_qwen2_tokenizer()
    models = generate_all_models()
    init_logs()

    print(f"\n{'='*70}\n🚀 VLLM INFERENCE — {len(models)} MODELS\n{'='*70}")
    print(f"📊 Samples per task:    {SAMPLES_PER_TASK}")
    print(f"📊 GPUs:                {TENSOR_PARALLEL}")
    print(f"📊 GPU memory util:     {GPU_MEMORY_UTIL}")

    results    = []
    start_time = time.time()

    for idx, model_config in enumerate(models, 1):
        print(f"\n{'='*70}\n📦 PROGRESS: {idx}/{len(models)}\n{'='*70}")
        result = run_inference(model_config)
        results.append(result)
        time.sleep(2)

    total_time  = time.time() - start_time
    successful  = sum(1 for r in results if r.get("status") == "success")

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"   Successful: {successful}/{len(models)}")
    print(f"   Total time: {total_time/3600:.2f} hours")
    print(f"   Results:    {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
