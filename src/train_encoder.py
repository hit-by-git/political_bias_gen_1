#!/usr/bin/env python3
"""
src/train_encoder.py

Fine-tunes transformer cross-encoders (MuRIL, IndicBERT, XLM-RoBERTa, DeBERTa)
for both:
1. Task A: Post-Only Classification (exposing affective shortcuts)
2. Task B: Evidence-Conditioned Claim Verification (measuring factual grounding)

Automatically evaluates on the test split, computes counterfactual metrics
(Macro-F1, Δ_ABG, TCR, Δ_CL), and exports results.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_hf_dataset(records: List[Dict[str, Any]], tokenizer, task: str, max_length: int):
    """Tokenize dataset for either post_only or evidence_conditioned task."""
    from datasets import Dataset

    texts_a = []
    texts_b = []
    labels = []

    for r in records:
        lbl = 1 if r["label"] == "misinfo" else 0
        labels.append(lbl)

        if task == "post_only":
            texts_a.append(r["post"])
            texts_b.append(None)
        elif task == "evidence_conditioned":
            texts_a.append(r["article"])
            texts_b.append(r["post"])
        else:
            raise ValueError(f"Unknown task: {task}. Choose 'post_only' or 'evidence_conditioned'.")

    if task == "post_only":
        encodings = tokenizer(
            texts_a,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )
    else:
        encodings = tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )

    encodings["label"] = labels
    return Dataset.from_dict(encodings)


def get_model_slug(model_name: str) -> str:
    """Creates a filesystem-safe slug from model name."""
    return model_name.strip("/").replace("/", "_").replace(".", "_").lower()


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
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            set_seed
        )
    except ImportError as e:
        print(f"❌ Error: Missing dependency: {e}. Install via: pip install torch transformers datasets accelerate")
        sys.exit(1)

    set_seed(args.seed)

    print(f"\n{'=' * 65}")
    print(f"🚀 INITIATING ENCODER TRAINING")
    print(f"• Model:      {args.model_name_or_path}")
    print(f"• Task:       {args.task}")
    print(f"• Epochs:     {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"• Max Length: {args.max_seq_length}")
    print(f"{'=' * 65}\n")

    # Load splits
    train_recs = load_jsonl(args.train_file)
    val_recs = load_jsonl(args.val_file)
    test_recs = load_jsonl(args.test_file)

    print(f"📂 Loaded: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label={0: "not_misinfo", 1: "misinfo"},
        label2id={"not_misinfo": 0, "misinfo": 1}
    )

    train_dataset = prepare_hf_dataset(train_recs, tokenizer, args.task, args.max_seq_length)
    val_dataset = prepare_hf_dataset(val_recs, tokenizer, args.task, args.max_seq_length)
    test_dataset = prepare_hf_dataset(test_recs, tokenizer, args.task, args.max_seq_length)

    slug = get_model_slug(args.model_name_or_path)
    run_name = f"{slug}_{args.task}"
    model_ckpt_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.results_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    
    # Ensure compatibility across transformers versions (v4.41+ uses eval_strategy)
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"

    training_kwargs = {
        "output_dir": model_ckpt_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.lr,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": args.fp16 and use_cuda,
        "logging_steps": 20,
        "save_total_limit": 1,
        "seed": args.seed,
        "report_to": "none"
    }
    training_args = TrainingArguments(**training_kwargs)

    def compute_metrics_fn(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds = preds.argmax(axis=-1)
        acc = (preds == eval_pred.label_ids).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn
    )

    # ── Train ──
    print("⏳ Starting model training...")
    trainer.train()

    # ── Test Evaluation & Counterfactual Inference ──
    print("\n🔍 Running inference on Test Split (489 posts)...")
    test_preds_raw = trainer.predict(test_dataset)
    logits = test_preds_raw.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_indices = probs.argmax(axis=-1)

    annotated_test_records = []
    for r, pred_idx, prob in zip(test_recs, pred_indices, probs):
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

    # Compute counterfactual metrics
    metrics = compute_all_metrics(annotated_test_records)
    print(format_text_report(metrics, model_name=f"{args.model_name_or_path} ({args.task})"))

    # Save summary json
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
