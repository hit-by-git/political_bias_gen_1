#!/usr/bin/env python3
"""
src/debiasing/contrastive_trainer.py

Implements Counterfactual Symmetrical Regularization (CDA) to eliminate
shortcut learning on affective slang and bias confounders:

L_total = L_CE + lambda_reg * MSE(h_neutral, h_hard_negative)

Forces the encoder to project neutral and emotionally-charged true posts
to near-identical representations in fact-space, drastically shrinking Δ_ABG.
"""

import os
import sys
import json
import argparse
import inspect
from collections import defaultdict
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
try:
    from src.evaluation.evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table
except ImportError:
    from evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table


def load_dataset_by_events(file_path: str):
    """Load records grouped by event_id."""
    events = defaultdict(dict)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                events[r["event_id"]][r["variant"]] = r
    return events


def main():
    parser = argparse.ArgumentParser(description="Contrastive Symmetrical Regularization Trainer.")
    parser.add_argument("--model_name_or_path", type=str, default="google/muril-base-cased")
    parser.add_argument("--train_file", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/splits/val.jsonl")
    parser.add_argument("--test_file", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints/contrastive_muril")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--lambda_reg", type=float, default=0.5, help="Weight for contrastive distance penalty")
    parser.add_argument("--batch_size", type=int, default=8, help="Events per batch")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
        from torch.optim.lr_scheduler import LambdaLR
    except ImportError as e:
        print(f"❌ Error: Missing dependency: {e}. Install via: pip install torch transformers")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"\n{'=' * 65}")
    print(f"🛡️ CONTRASTIVE SYMMETRICAL DEBIASING (CDA)")
    print(f"• Model:      {args.model_name_or_path}")
    print(f"• Device:     {device}")
    print(f"• Lambda:     {args.lambda_reg} (Distance penalty between Neutral and Hard Negative)")
    print(f"• Epochs:     {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"{'=' * 65}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        output_hidden_states=True
    ).to(device)

    train_events = load_dataset_by_events(args.train_file)
    is_quadruplet = any(
        ("neutral_factual" in ev and "toxic_factual" in ev)
        for ev in train_events.values()
    )

    if is_quadruplet:
        event_list = [
            ev for ev in train_events.values()
            if "neutral_factual" in ev and "toxic_factual" in ev and ("neutral_misinfo" in ev or "toxic_misinfo" in ev)
        ]
        format_name = "2x2 Factorial Quadruplets"
    else:
        event_list = [
            ev for ev in train_events.values()
            if "neutral" in ev and "hard_negative" in ev and "misinfo" in ev
        ]
        format_name = "Contrastive Triplets"

    print(f"📂 Loaded {len(event_list)} complete {format_name} for contrastive learning.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(event_list) * args.epochs // args.batch_size)
    num_warmup_steps = int(total_steps * 0.1)
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - num_warmup_steps)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # Cache model forward parameter names to filter tokenizer outputs
    sig = inspect.signature(model.forward)
    model_param_names = set(sig.parameters.keys())

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0

        for i in range(0, len(event_list), args.batch_size):
            batch = event_list[i:i + args.batch_size]

            # Flatten variants into batch
            texts = []
            labels = []
            neutral_idx = []
            hard_neg_idx = []

            for item_idx, ev in enumerate(batch):
                curr_base = len(texts)
                if is_quadruplet:
                    # neutral factual
                    texts.append((ev["neutral_factual"]["article"], ev["neutral_factual"]["post"]))
                    labels.append(0)
                    neutral_idx.append(curr_base)

                    # toxic factual
                    texts.append((ev["toxic_factual"]["article"], ev["toxic_factual"]["post"]))
                    labels.append(0)
                    hard_neg_idx.append(curr_base + 1)

                    # misinfo variants
                    if "neutral_misinfo" in ev:
                        texts.append((ev["neutral_misinfo"]["article"], ev["neutral_misinfo"]["post"]))
                        labels.append(1)
                    if "toxic_misinfo" in ev:
                        texts.append((ev["toxic_misinfo"]["article"], ev["toxic_misinfo"]["post"]))
                        labels.append(1)
                else:
                    # neutral
                    texts.append((ev["neutral"]["article"], ev["neutral"]["post"]))
                    labels.append(0)
                    neutral_idx.append(curr_base)

                    # hard_negative
                    texts.append((ev["hard_negative"]["article"], ev["hard_negative"]["post"]))
                    labels.append(0)
                    hard_neg_idx.append(curr_base + 1)

                    # misinfo
                    texts.append((ev["misinfo"]["article"], ev["misinfo"]["post"]))
                    labels.append(1)

            inputs = tokenizer(
                [t[0] for t in texts],
                [t[1] for t in texts],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            targets = torch.tensor(labels, dtype=torch.long).to(device)

            # Filter model inputs to only include supported parameters
            model_inputs = {k: v for k, v in inputs.items() if k in model_param_names and k != "labels"}

            optimizer.zero_grad()
            outputs = model(**model_inputs, labels=targets)
            ce_loss = outputs.loss
            
            # Extract last hidden states (pooled representation of [CLS] token)
            last_hidden = outputs.hidden_states[-1][:, 0, :]
            h_neutral = last_hidden[neutral_idx]
            h_hard_neg = last_hidden[hard_neg_idx]

            # Symmetrical representation distance penalty
            reg_loss = mse_loss_fn(h_neutral, h_hard_neg)
            loss = ce_loss + args.lambda_reg * reg_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_reg += reg_loss.item()

        avg_loss = total_loss / max(1, (len(event_list) // args.batch_size))
        print(f"Epoch {epoch + 1}/{args.epochs} — Loss: {avg_loss:.4f} (CE: {total_ce / max(1, len(event_list) // args.batch_size):.4f}, Reg: {total_reg / max(1, len(event_list) // args.batch_size):.4f})")

    # ── Save Debiased Checkpoint ──
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"💾 Saved debiased model checkpoint to {args.output_dir}")

    # ── Test Evaluation ──
    print("\n🔍 Evaluating debiased model on Test Split...")
    test_recs = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_recs.append(json.loads(line))

    model.eval()
    annotated = []
    with torch.no_grad():
        for i in range(0, len(test_recs), 16):
            chunk = test_recs[i:i + 16]
            inputs = tokenizer(
                [r["article"] for r in chunk],
                [r["post"] for r in chunk],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            model_inputs = {k: v for k, v in inputs.items() if k in model_param_names and k != "labels"}
            logits = model(**model_inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for r, pred in zip(chunk, preds):
                item = dict(r)
                item["pred_label"] = "misinfo" if pred == 1 else "not_misinfo"
                annotated.append(item)

    metrics = compute_all_metrics(annotated)
    print(format_text_report(metrics, model_name=f"{args.model_name_or_path} + CDA Debiasing"))

    os.makedirs(args.results_dir, exist_ok=True)
    out_preds = os.path.join(args.results_dir, "debiased_muril_test_preds.jsonl")
    with open(out_preds, "w", encoding="utf-8") as f:
        for r in annotated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    latex_file = os.path.join(args.results_dir, "debiased_muril_table.tex")
    latex_code = export_latex_table([{
        "model_name": "MuRIL + CDA (Ours)",
        "task": "Evidence-Conditioned",
        "metrics": metrics
    }])
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_code)

    summary_file = os.path.join(args.results_dir, "debiased_muril_metrics.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Symmetrical Debiasing complete! Saved predictions to {out_preds}")


if __name__ == "__main__":
    main()
