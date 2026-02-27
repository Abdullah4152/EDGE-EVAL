"""
compute_all_metrics.py
-----------------------
Orchestrates computation of all five EdgeEval metrics:
  1. Q_ret  — Quantization Fidelity
  2. N_break — Economic Break-Even
  3. IPW    — Intelligence Per Watt
  4. rho_sys — System Density
  5. C_tax  — Cold-Start Tax

Loads all required CSVs, runs each metric function, saves results,
and produces a combined summary report.

Usage:
    python src/metrics/compute_all_metrics.py \
        --data_dir   ./results \
        --output_dir ./results

All intermediate CSVs and the summary .txt are written to output_dir.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from q_ret   import calculate_quantization_fidelity
from n_break import calculate_economic_breakeven
from ipw     import calculate_intelligence_per_watt
from rho_sys import calculate_system_density
from c_tax   import calculate_cold_start_tax

# ─────────────────────────────────────────────────────────
# Constants (adjustable via CLI if needed)
# ─────────────────────────────────────────────────────────
GPT4O_COST_PER_1K_TOKENS = 0.0025      # $2.50 per 1M tokens (GPT-4o, as of paper)
AVG_TOKENS_PER_REQUEST   = 500
GPT4O_COST_PER_REQUEST   = (AVG_TOKENS_PER_REQUEST / 1000) * GPT4O_COST_PER_1K_TOKENS
COST_PER_KWH             = 0.12        # USD, US average electricity price

MODEL_SIZES_GB = {
    ("1B", "FP16"): 2.0,
    ("1B", "INT8"): 1.0,
    ("1B", "INT4"): 0.625,
    ("3B", "FP16"): 6.0,
    ("3B", "INT8"): 3.0,
    ("3B", "INT4"): 1.875,
    ("7B", "FP16"): 14.0,
    ("7B", "INT8"): 7.0,
    ("7B", "INT4"): 4.375,
}


def load_data(data_dir: Path) -> dict:
    print("Loading datasets...")

    llama_perf  = pd.read_csv(data_dir / "final_36_variants_report_llama.csv")
    qwen_perf   = pd.read_csv(data_dir / "final_36_variants_report_qwen.csv")
    inference   = pd.read_csv(data_dir / "vllm_inference_metrics_tesla_t4_dual.csv")
    llama_train = pd.read_csv(data_dir / "training_energy_log_llama.csv")
    qwen_train  = pd.read_csv(data_dir / "training_energy_log_qwen.csv")

    print(f"  ✓ {len(llama_perf)}  Llama performance variants")
    print(f"  ✓ {len(qwen_perf)}   Qwen performance variants")
    print(f"  ✓ {len(inference)}   inference records")
    print(f"  ✓ {len(llama_train)} Llama training records")
    print(f"  ✓ {len(qwen_train)}  Qwen training records")

    # Split inference by family
    llama_inf = inference[inference["Family"].str.lower() == "llama"].copy()
    qwen_inf  = inference[inference["Family"].str.lower() == "qwen"].copy()

    return {
        "llama_perf":  llama_perf,
        "qwen_perf":   qwen_perf,
        "llama_inf":   llama_inf,
        "qwen_inf":    qwen_inf,
        "llama_train": llama_train,
        "qwen_train":  qwen_train,
        "inference":   inference,
    }


def run_all(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)

    results = {}

    # ── 1. Quantization Fidelity ──────────────────────────────────────────
    print("\n" + "="*70)
    print("METRIC 1: Quantization Fidelity (Q_ret)")
    print("="*70)
    results["llama_q_ret"] = calculate_quantization_fidelity(data["llama_perf"].copy(), "Llama")
    results["qwen_q_ret"]  = calculate_quantization_fidelity(data["qwen_perf"].copy(),  "Qwen")

    q_ret_combined = pd.concat([results["llama_q_ret"], results["qwen_q_ret"]], ignore_index=True)
    q_ret_combined.to_csv(output_dir / "q_ret.csv", index=False)
    print(f"  ✓ Saved: q_ret.csv ({len(q_ret_combined)} rows)")

    # ── 2. Economic Break-Even ────────────────────────────────────────────
    print("\n" + "="*70)
    print("METRIC 2: Economic Break-Even (N_break)")
    print("="*70)
    results["llama_n_break"] = calculate_economic_breakeven(
        data["llama_train"], data["llama_inf"], "Llama",
        GPT4O_COST_PER_REQUEST, COST_PER_KWH,
    )
    results["qwen_n_break"] = calculate_economic_breakeven(
        data["qwen_train"], data["qwen_inf"], "Qwen",
        GPT4O_COST_PER_REQUEST, COST_PER_KWH,
    )

    n_break_combined = pd.concat([results["llama_n_break"], results["qwen_n_break"]], ignore_index=True)
    n_break_combined.to_csv(output_dir / "n_break.csv", index=False)
    print(f"  ✓ Saved: n_break.csv ({len(n_break_combined)} rows)")

    # ── 3. Intelligence Per Watt ──────────────────────────────────────────
    print("\n" + "="*70)
    print("METRIC 3: Intelligence Per Watt (IPW)")
    print("="*70)
    results["llama_ipw"] = calculate_intelligence_per_watt(
        data["llama_perf"].copy(), data["llama_inf"], "Llama")
    results["qwen_ipw"] = calculate_intelligence_per_watt(
        data["qwen_perf"].copy(),  data["qwen_inf"],  "Qwen")

    ipw_combined = pd.concat([results["llama_ipw"], results["qwen_ipw"]], ignore_index=True)
    ipw_combined.to_csv(output_dir / "ipw.csv", index=False)
    print(f"  ✓ Saved: ipw.csv ({len(ipw_combined)} rows)")

    # ── 4. System Density ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("METRIC 4: System Density (ρ_sys)")
    print("="*70)
    results["llama_rho_sys"] = calculate_system_density(data["llama_inf"], "Llama", MODEL_SIZES_GB)
    results["qwen_rho_sys"]  = calculate_system_density(data["qwen_inf"],  "Qwen",  MODEL_SIZES_GB)

    rho_combined = pd.concat([results["llama_rho_sys"], results["qwen_rho_sys"]], ignore_index=True)
    rho_combined.to_csv(output_dir / "rho_sys.csv", index=False)
    print(f"  ✓ Saved: rho_sys.csv ({len(rho_combined)} rows)")

    # ── 5. Cold-Start Tax ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("METRIC 5: Cold-Start Tax (C_tax)")
    print("="*70)
    results["llama_c_tax"] = calculate_cold_start_tax(data["llama_inf"], "Llama")
    results["qwen_c_tax"]  = calculate_cold_start_tax(data["qwen_inf"],  "Qwen")

    c_tax_combined = pd.concat([results["llama_c_tax"], results["qwen_c_tax"]], ignore_index=True)
    c_tax_combined.to_csv(output_dir / "c_tax.csv", index=False)
    print(f"  ✓ Saved: c_tax.csv ({len(c_tax_combined)} rows)")

    # ── Summary report ────────────────────────────────────────────────────
    from summary_report import generate_summary_report, create_combined_metrics
    generate_summary_report(results, output_dir)
    create_combined_metrics(results, output_dir)

    print("\n" + "="*70)
    print("✅ ALL METRICS COMPUTED")
    print(f"   Output directory: {output_dir}")
    print("="*70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute all 5 EdgeEval metrics")
    parser.add_argument("--data_dir",   type=Path, default=Path("./results"),
                        help="Directory containing input CSVs")
    parser.add_argument("--output_dir", type=Path, default=Path("./results"),
                        help="Directory to write output CSVs and report")
    args = parser.parse_args()
    run_all(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
