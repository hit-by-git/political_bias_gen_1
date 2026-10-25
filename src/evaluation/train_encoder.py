#!/usr/bin/env python3
"""
src/train_encoder.py

Fine-tunes transformer cross-encoders (MuRIL, IndicBERT, XLM-RoBERTa, DeBERTa)
using native PyTorch with mixed precision (AMP) and linear warmup.

Does NOT depend on HuggingFace `Trainer` or `accelerate`, preventing any
dependency conflict or lazy-import errors on Kaggle / Colab.

Supports:
1. Task A: Post-Only Classification (exposing affective shortcuts)
2. Task B: Evidence-Conditioned Claim Verification (measuring factual grounding)

Automatically evaluates on the test split, computes counterfactual metrics
(Macro-F1, Δ_ABG, TCR, Δ_CL), and exports Overleaf LaTeX tables.
"""

import os
import sys
import json
import argparse
import inspect
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from src.evaluation.evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table
except ImportError:
    from evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_model_slug(model_name: str) -> str:
    """Creates a filesystem-safe slug from model name."""
    return model_name.strip("/").replace("/", "_").replace(".", "_").lower()


class PostDataset:
    """Lightweight PyTorch Dataset for single and paired texts."""
    def __init__(self, records: List[Dict[str, Any]], task: str):
        self.records = records
        self.task = task

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        lbl = 1 if r["label"] == "misinfo" else 0
        if self.task == "post_only":
            return r["post"], None, lbl
        else:
            return r["article"], r["post"], lbl


def make_collate_fn(tokenizer, max_length: int, task: str):
    import torch
    def collate_fn(batch):
        texts_a = [item[0] for item in batch]
        texts_b = [item[1] for item in batch] if task == "evidence_conditioned" else None
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

        if texts_b is not None:
            inputs = tokenizer(
                texts_a,
                texts_b,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        else:
            inputs = tokenizer(
                texts_a,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

        inputs["labels"] = labels
        return inputs
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Fine-tune encoder for misinformation verification.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/muril-base-cased",
        help="HuggingFace model ID (e.g., google/muril-base-cased, ai4bharat/indic-bert, xlm-roberta-base, microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["post_only", "evidence_conditioned"],
        default="evidence_conditioned",
        help="Task formulation: 'post_only' or 'evidence_conditioned'"
    )
    parser.add_argument("--train_file", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/splits/val.jsonl")
    parser.add_argument("--test_file", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision on CUDA")
    args = parser.parse_args()

    # Dynamic default sequence length
    if args.max_seq_length is None:
        args.max_seq_length = 256 if args.task == "post_only" else 512

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            set_seed
        )
        from torch.optim.lr_scheduler import LambdaLR
    except ImportError as e:
        print(f"❌ Error: Missing dependency ({e}). Install via: pip install torch transformers")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_cuda = (device.type == "cuda")

    print(f"\n{'=' * 65}")
    print(f"🚀 INITIATING ENCODER TRAINING (PyTorch Native)")
    print(f"• Model:      {args.model_name_or_path}")
    print(f"• Device:     {device} ({torch.cuda.get_device_name(0) if use_cuda else device.type.upper()})")
    print(f"• Task:       {args.task}")
    print(f"• Epochs:     {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"• Max Length: {args.max_seq_length} | FP16: {args.fp16 and use_cuda}")
    print(f"{'=' * 65}\n")

    # Load data splits
    train_recs = load_jsonl(args.train_file)
    val_recs = load_jsonl(args.val_file)
    test_recs = load_jsonl(args.test_file)

    print(f"📂 Loaded: Train={len(train_recs)} posts, Val={len(val_recs)} posts, Test={len(test_recs)} posts")

    # Initialize Tokenizer & Model
    print("⏳ Loading tokenizer and model weights...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label={0: "not_misinfo", 1: "misinfo"},
        label2id={"not_misinfo": 0, "misinfo": 1}
    ).to(device)

    # Datasets and DataLoaders
    collate_fn = make_collate_fn(tokenizer, args.max_seq_length, args.task)
    train_loader = DataLoader(
        PostDataset(train_recs, args.task),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        PostDataset(val_recs, args.task),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        PostDataset(test_recs, args.task),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    slug = get_model_slug(args.model_name_or_path)
    run_name = f"{slug}_{args.task}"
    model_ckpt_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(total_steps * 0.1)
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - num_warmup_steps)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.fp16 and use_cuda))

    # Inspect model forward signature for input filtering
    sig = inspect.signature(model.forward)
    model_param_names = set(sig.parameters.keys())

    # ── Training Loop ──
    best_val_loss = float("inf")
    print(f"⏳ Training across {args.epochs} epochs ({len(train_loader)} batches/epoch)...")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k in model_param_names and k != "labels"}
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=(args.fp16 and use_cuda)):
                outputs = model(**batch_inputs, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            if batch_idx % 40 == 0 or batch_idx == len(train_loader):
                print(f"  [Epoch {epoch + 1}/{args.epochs}] Batch {batch_idx:3d}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation Loop ──
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_inputs = {k: v.to(device) for k, v in batch.items() if k in model_param_names and k != "labels"}
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device.type, enabled=(args.fp16 and use_cuda)):
                    outputs = model(**batch_inputs, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                total_val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        avg_val_loss = total_val_loss / max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100

        print(f"✅ Epoch {epoch + 1}/{args.epochs} Complete | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(model_ckpt_dir)
            tokenizer.save_pretrained(model_ckpt_dir)

    # ── Test Evaluation & Counterfactual Inference ──
    print("\n🔍 Evaluating best model on Test Split (489 posts)...")
    # Load best model weights if saved
    if os.path.exists(os.path.join(model_ckpt_dir, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt_dir).to(device)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k in model_param_names and k != "labels"}
            with torch.amp.autocast(device.type, enabled=(args.fp16 and use_cuda)):
                logits = model(**batch_inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)

    annotated_test_records = []
    for r, prob in zip(test_recs, all_probs):
        pred_idx = prob.argmax()
        item = dict(r)
        item["pred_label"] = "misinfo" if pred_idx == 1 else "not_misinfo"
        item["prob_misinfo"] = float(prob[1])
        item["prob_not_misinfo"] = float(prob[0])
        annotated_test_records.append(item)

    # Save detailed test predictions
    test_preds_file = os.path.join(args.results_dir, f"{run_name}_test_preds.jsonl")
    with open(test_preds_file, "w", encoding="utf-8") as f:
        for r in annotated_test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute thesis counterfactual metrics
    metrics = compute_all_metrics(annotated_test_records)
    print(format_text_report(metrics, model_name=f"{args.model_name_or_path} ({args.task})"))

    # Save summary metrics json
    summary_file = os.path.join(args.results_dir, f"{run_name}_metrics.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Export LaTeX table row
    latex_file = os.path.join(args.results_dir, f"{run_name}_table.tex")
    latex_code = export_latex_table([{
        "model_name": args.model_name_or_path.split("/")[-1],
        "task": args.task,
        "metrics": metrics
    }])
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"✅ Run complete!")
    print(f"   • Predictions: {test_preds_file}")
    print(f"   • Metrics:     {summary_file}")
    print(f"   • LaTeX:       {latex_file}")


if __name__ == "__main__":
    main()
