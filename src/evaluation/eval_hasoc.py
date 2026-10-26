#!/usr/bin/env python3
"""
src/evaluation/eval_hasoc.py

Step 2 of the 3-Step Evaluation Protocol:
Cross-Task Toxicity Confounding Audit on HASOC 2019 Benchmark.

Tests whether:
1. Misinformation detectors falsely flag hostile/offensive language (HOF) as misinformation.
2. The debiased model reduces false-positive misclassification on vernacular abusive/hostile language.
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from src.evaluation.evaluate_metrics import compute_binary_metrics, export_latex_table
except ImportError:
    from evaluate_metrics import compute_binary_metrics, export_latex_table


def load_hasoc_test(tsv_path: str) -> pd.DataFrame:
    """Load HASOC 2019 English test set."""
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"HASOC test set not found at: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", header=0)
    # Expected columns: text_id, text, task_1 (NOT / HOF)
    if "task_1" not in df.columns or "text" not in df.columns:
        # Fallback to column indices if headers differ
        df.columns = ["text_id", "text", "task_1", "task_2", "task_3"][:len(df.columns)]
    return df


def main():
    parser = argparse.ArgumentParser(description="Step 2: HASOC 2019 Toxicity Confounding Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/checkpoints/contrastive_muril",
        help="Path to trained model checkpoint or HF model name"
    )
    parser.add_argument(
        "--hasoc_test_file",
        type=str,
        default="data/HASOC_eng/hasoc2019_en_test-2919.tsv",
        help="Path to HASOC 2019 TSV test file"
    )
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        print(f"❌ Missing dependency: {e}. Install via pip install torch transformers")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device)

    print(f"\n{'=' * 65}")
    print(f"🛡️ STEP 2 EVALUATION: HASOC 2019 TOXICITY AUDIT")
    print(f"• Model:      {args.model_path}")
    print(f"• Test Set:   {args.hasoc_test_file}")
    print(f"• Device:     {device}")
    print(f"{'=' * 65}\n")

    df = load_hasoc_test(args.hasoc_test_file)
    print(f"📂 Loaded {len(df)} HASOC test posts.")
    print(f"   Distribution: {dict(df['task_1'].value_counts())}")

    # Load tokenizer and model
    print("⏳ Loading model weights...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    texts = df["text"].astype(str).tolist()
    hasoc_labels = df["task_1"].astype(str).tolist()

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            ).to(device)

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)

    # Analyze how the misinformation detector treats HOF vs NOT
    # In model: 1 = misinfo, 0 = not_misinfo
    results = []
    for text_id, text, hasoc_lbl, pred_idx, prob in zip(df["text_id"], texts, hasoc_labels, all_preds, all_probs):
        pred_label = "misinfo" if pred_idx == 1 else "not_misinfo"
        results.append({
            "text_id": str(text_id),
            "text": text,
            "hasoc_label": hasoc_lbl,
            "pred_misinfo_label": pred_label,
            "prob_misinfo": float(prob[1]),
            "prob_not_misinfo": float(prob[0])
        })

    # Compute False Positive Rate on Hate Speech (HOF flagged as Misinformation)
    hof_records = [r for r in results if r["hasoc_label"] == "HOF"]
    not_records = [r for r in results if r["hasoc_label"] == "NOT"]

    hof_flagged_misinfo = sum(1 for r in hof_records if r["pred_misinfo_label"] == "misinfo")
    not_flagged_misinfo = sum(1 for r in not_records if r["pred_misinfo_label"] == "misinfo")

    hof_fp_rate = (hof_flagged_misinfo / max(1, len(hof_records))) * 100
    not_fp_rate = (not_flagged_misinfo / max(1, len(not_records))) * 100
    toxicity_bias_gap = round(hof_fp_rate - not_fp_rate, 2)

    print(f"\n{'=' * 65}")
    print(f"📊 HASOC TOXICITY CONFOUNDING AUDIT RESULTS:")
    print(f"{'=' * 65}")
    print(f"• Total HOF Posts (Hate/Offensive): {len(hof_records)}")
    print(f"  - Flagged as Misinformation:      {hof_flagged_misinfo} ({hof_fp_rate:.2f}%)")
    print(f"• Total NOT Posts (Non-Offensive):  {len(not_records)}")
    print(f"  - Flagged as Misinformation:      {not_flagged_misinfo} ({not_fp_rate:.2f}%)")
    print(f"• Toxicity-to-Misinfo Confound Gap: {toxicity_bias_gap:+.2f}%")
    print(f"  (Lower gap indicates model is invariant to abusive sentiment)")
    print(f"{'=' * 65}\n")

    os.makedirs(args.results_dir, exist_ok=True)
    slug = args.model_path.strip("/").split("/")[-1].replace(".", "_").lower()
    out_preds = os.path.join(args.results_dir, f"{slug}_hasoc_audit_preds.jsonl")
    with open(out_preds, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model_path,
        "total_posts": len(results),
        "total_hof": len(hof_records),
        "total_not": len(not_records),
        "hof_flagged_misinfo": hof_flagged_misinfo,
        "hof_flagged_misinfo_rate": round(hof_fp_rate, 2),
        "not_flagged_misinfo": not_flagged_misinfo,
        "not_flagged_misinfo_rate": round(not_fp_rate, 2),
        "toxicity_bias_gap": toxicity_bias_gap
    }

    out_json = os.path.join(args.results_dir, f"{slug}_hasoc_audit_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Step 2 Complete! Saved audit predictions to {out_preds}")
    print(f"   Summary JSON: {out_json}")


if __name__ == "__main__":
    main()
