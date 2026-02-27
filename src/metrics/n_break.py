"""
n_break.py
----------
Metric 2: Economic Break-Even (N_break)

N_break = C_train / (C_api_per_req - C_local_per_req)

Estimates the number of inference requests needed before self-hosting
a fine-tuned model becomes cheaper than paying GPT-4o API costs.

C_train        = training energy (Joules → kWh → USD)
C_api_per_req  = GPT-4o cost for a single request (~500 tokens)
C_local_per_req= energy cost of one local inference request

ROI_Within_Hour = N_break < 20 requests (i.e., breaks even within ~1 hour
                   of production traffic at ~20 req/min)
"""

import math
import pandas as pd


def calculate_economic_breakeven(
    train_df:          pd.DataFrame,
    inference_df:      pd.DataFrame,
    family_name:       str,
    gpt4o_cost_per_req: float,
    cost_per_kwh:      float,
) -> pd.DataFrame:
    """
    Args:
        train_df:           training_energy_log_{family}.csv
        inference_df:       vllm_inference_metrics filtered to this family
        family_name:        "Llama" or "Qwen"
        gpt4o_cost_per_req: USD cost per GPT-4o call (e.g. 0.00125)
        cost_per_kwh:       USD per kWh (e.g. 0.12)

    Returns:
        DataFrame with N_break and cost details for every model variant.
    """
    print(f"\n{'='*80}")
    print(f"METRIC 2: Economic Break-Even (N_break) — {family_name}")
    print(f"{'='*80}")

    train_df = train_df.copy()
    # Normalise method names
    train_df["Method_Clean"] = (
        train_df["Method"]
        .str.replace("_FP16", "", regex=False)
        .str.replace("_INT4", "", regex=False)
    )

    results = []

    for size in ["1B", "3B", "7B"]:
        for method in ["LoRA", "QLoRA"]:
            for task in ["rag", "summ", "chat"]:

                # Training cost
                t_mask = (
                    (train_df["Size"] == size) &
                    (train_df["Method_Clean"] == method) &
                    (train_df["Task"] == task)
                )
                t_sub = train_df[t_mask]
                if len(t_sub) == 0:
                    continue

                training_joules = t_sub["Joules"].values[0]
                training_kwh    = training_joules / 3.6e6
                training_cost   = training_kwh * cost_per_kwh

                # Inference cost per request for each quantization
                for quant in ["FP16", "INT8", "INT4"]:
                    i_mask = (
                        (inference_df["Size"] == size) &
                        (inference_df["Method"] == method) &
                        (inference_df["Task"] == task) &
                        (inference_df["Quantization"] == quant)
                    )
                    i_sub = inference_df[i_mask]
                    if len(i_sub) == 0:
                        continue

                    total_energy_j = i_sub["Energy_Total_J"].values[0]
                    num_samples    = i_sub["Num_Samples"].values[0]
                    e_per_req_j    = total_energy_j / num_samples
                    e_per_req_kwh  = e_per_req_j / 3.6e6
                    local_cost     = e_per_req_kwh * cost_per_kwh

                    cost_diff = gpt4o_cost_per_req - local_cost
                    n_break   = (training_cost / cost_diff) if cost_diff > 0 else math.inf

                    results.append({
                        "Family":               family_name,
                        "Size":                 size,
                        "Method":               method,
                        "Task":                 task,
                        "Quantization":         quant,
                        "Model_Name":           i_sub["Model_ID"].values[0],
                        "Training_Cost_USD":    training_cost,
                        "Training_Energy_kWh":  training_kwh,
                        "GPT4o_Cost_Per_Req":   gpt4o_cost_per_req,
                        "Local_Cost_Per_Req":   local_cost,
                        "Cost_Savings_Per_Req": cost_diff,
                        "N_break":              n_break,
                        "ROI_Within_Hour":      "Yes" if n_break < 20 else "No",
                        "Hours_to_Breakeven":   n_break / 60 if n_break != math.inf else math.inf,
                    })

    df = pd.DataFrame(results)

    if len(df) > 0:
        finite = df[df["N_break"] != math.inf]
        print(f"\n  Total variants analysed: {len(df)}")
        if len(finite):
            print(f"\n  Break-even Statistics:")
            print(f"    Mean  N_break: {finite['N_break'].mean():.0f} requests")
            print(f"    Median N_break: {finite['N_break'].median():.0f} requests")
            for q in ["INT4", "INT8", "FP16"]:
                sub = finite[finite["Quantization"] == q]
                if len(sub):
                    print(f"    {q}: Mean={sub['N_break'].mean():.0f} req  "
                          f"({sub['N_break'].mean()/60:.1f} h)")

    return df
