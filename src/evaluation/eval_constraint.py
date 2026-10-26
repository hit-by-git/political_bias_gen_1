#!/usr/bin/env python3
"""
src/evaluation/eval_constraint.py

Step 3 of the 3-Step Evaluation Protocol:
Real-World Out-of-Domain Generalization Benchmark on Constraint@AAAI 2021.

Evaluates whether debiased and baseline encoders generalize to real-world,
messy Indian social media posts verified by fact-checkers (AltNews, BoomLive).
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


def main():
    parser = argparse.ArgumentParser(description="Step 3: Constraint@AAAI 2021 Generalization Benchmark")
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/checkpoints/contrastive_muril",
        help="Path to trained model checkpoint or HF model name"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/constraint_aaai2021/test.csv",
        help="Path to Constraint@AAAI 2021 test CSV file"
    )
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        print(f"❌ Missing dependency: {e}. Install via pip install torch transformers")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device)
    use_cuda = (device.type == "cuda")

    print(f"\n{'=' * 65}")
    print(f"🌐 STEP 3 EVALUATION: CONSTRAINT@AAAI 2021 GENERALIZATION")
    print(f"• Model:      {args.model_path}")
    print(f"• Test Set:   {args.test_file}")
    print(f"• Device:     {device}")
    print(f"{'=' * 65}\n")

    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Constraint test file not found at: {args.test_file}")

    df = pd.read_csv(args.test_file)
    print(f"📂 Loaded {len(df)} real-world Constraint@AAAI test posts.")
    print(f"   Distribution: {dict(df['label'].value_counts())}")

    # Load tokenizer and model
    print("⏳ Loading model weights...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    texts = df["tweet"].astype(str).tolist()
    # In Constraint: 'fake' = 1 (misinfo), 'real' = 0 (not_misinfo)
    ground_truth = [1 if str(l).strip().lower() == "fake" else 0 for l in df["label"]]

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

            with torch.amp.autocast(device.type, enabled=(args.fp16 and use_cuda)):
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)

    # Compute standard classification metrics
    metrics = compute_binary_metrics(ground_truth, all_preds)

    print(f"\n{'=' * 65}")
    print(f"🎯 CONSTRAINT@AAAI 2021 EMPIRICAL RESULTS:")
    print(f"{'=' * 65}")
    print(f"• Overall Accuracy:    {metrics['accuracy']:.2f}%")
    print(f"• Macro-F1:            {metrics['macro_f1']:.2f}%")
    print(f"• Fake (Misinfo) F1:   {metrics['misinfo_f1']:.2f}%")
    print(f"• Precision:           {metrics['misinfo_precision']:.2f}%")
    print(f"• Recall:              {metrics['misinfo_recall']:.2f}%")
    print(f"• Confusion Matrix:    TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    print(f"{'=' * 65}\n")

    os.makedirs(args.results_dir, exist_ok=True)
    slug = args.model_path.strip("/").split("/")[-1].replace(".", "_").lower()

    # Save detailed predictions
    annotated = []
    for tweet_id, text, true_lbl, pred_idx, prob in zip(df["id"], texts, df["label"], all_preds, all_probs):
        annotated.append({
            "id": str(tweet_id),
            "tweet": text,
            "true_label": true_lbl,
            "pred_label": "fake" if pred_idx == 1 else "real",
            "prob_fake": float(prob[1]),
            "prob_real": float(prob[0])
        })

    out_preds = os.path.join(args.results_dir, f"{slug}_constraint_preds.jsonl")
    with open(out_preds, "w", encoding="utf-8") as f:
        for r in annotated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary metrics
    out_json = os.path.join(args.results_dir, f"{slug}_constraint_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Export LaTeX table row
    latex_file = os.path.join(args.results_dir, f"{slug}_constraint_table.tex")
    latex_row = (
        f"{args.model_path.split('/')[-1]} & Constraint@AAAI 2021 & "
        f"{metrics['macro_f1']:.1f} & {metrics['accuracy']:.1f} & "
        f"{metrics['misinfo_precision']:.1f} & {metrics['misinfo_recall']:.1f} & "
        f"{metrics['misinfo_f1']:.1f} \\\\\n"
    )
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_row)

    print(f"✅ Step 3 Complete! Saved predictions to {out_preds}")
    print(f"   Summary JSON: {out_json}")
    print(f"   LaTeX Row:    {latex_file}")


if __name__ == "__main__":
    main()
