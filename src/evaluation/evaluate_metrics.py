#!/usr/bin/env python3
"""
src/evaluate_metrics.py

Comprehensive evaluation suite for Master's thesis benchmarks:
- Standard Classification: Macro-F1, Binary F1, Precision, Recall, Accuracy
- Affective Bias Gap (Δ_ABG): Accuracy(Neutral) - Accuracy(Hard Negative)
- Triplet Consistency Rate (TCR): Strict 3-variant joint consistency per event
- Cross-Lingual Gap (Δ_CL): Performance drop from English to Hinglish/Hindi
- Mutation Tactic Breakdown: Accuracy per perturbation type
- Automated LaTeX Table Generation for Overleaf / Thesis chapters
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Computes precision, recall, f1 for binary classification.
    1 = misinfo, 0 = not_misinfo
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    # Macro F1 (average of misinfo and not_misinfo F1)
    prec_0 = tn / max(1, tn + fn)
    rec_0 = tn / max(1, tn + fp)
    f1_0 = 2 * prec_0 * rec_0 / max(1e-9, prec_0 + rec_0)
    macro_f1 = (f1 + f1_0) / 2.0

    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "misinfo_f1": round(f1 * 100, 2),
        "misinfo_precision": round(prec * 100, 2),
        "misinfo_recall": round(rec * 100, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": len(y_true)
    }


def compute_all_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes all standard and counterfactual metrics for records containing:
    'label' (ground truth: 'misinfo' or 'not_misinfo'),
    'pred_label' (predicted: 'misinfo' or 'not_misinfo'),
    'variant' ('neutral', 'hard_negative', 'misinfo'),
    'event_id', 'language', 'tactic'.
    """
    y_true = [1 if r["label"] == "misinfo" else 0 for r in records]
    y_pred = [1 if r["pred_label"] == "misinfo" else 0 for r in records]

    overall = compute_binary_metrics(y_true, y_pred)

    # ── Variant-Specific Performance ──
    variant_records = defaultdict(list)
    for r in records:
        variant_records[r.get("variant", "unknown")].append(r)

    variant_metrics = {}
    for var, v_recs in variant_records.items():
        vt = [1 if r["label"] == "misinfo" else 0 for r in v_recs]
        vp = [1 if r["pred_label"] == "misinfo" else 0 for r in v_recs]
        variant_metrics[var] = compute_binary_metrics(vt, vp)

    is_quadruplet = any(v in variant_metrics for v in ["neutral_factual", "toxic_factual", "neutral_misinfo", "toxic_misinfo"])

    # ── 1. Affective Bias Gap (Δ_ABG) ──
    # Δ_ABG = Acc(Neutral True) - Acc(Hostile/Slang True)
    if is_quadruplet:
        acc_neutral = variant_metrics.get("neutral_factual", {}).get("accuracy", 0.0)
        acc_hard_neg = variant_metrics.get("toxic_factual", {}).get("accuracy", 0.0)
    else:
        acc_neutral = variant_metrics.get("neutral", {}).get("accuracy", 0.0)
        acc_hard_neg = variant_metrics.get("hard_negative", {}).get("accuracy", 0.0)
    delta_abg = round(acc_neutral - acc_hard_neg, 2)

    # ── 2. Consistency Rate (TCR / QCR) ──
    # Fraction of events where ALL variants are predicted correctly
    events = defaultdict(dict)
    for r in records:
        ev_id = r.get("event_id")
        var = r.get("variant")
        is_correct = (r["label"] == r["pred_label"])
        events[ev_id][var] = is_correct

    total_events = len(events)
    consistent_events = 0
    for ev_id, var_dict in events.items():
        if is_quadruplet:
            if (
                var_dict.get("neutral_factual", False) and
                var_dict.get("toxic_factual", False) and
                var_dict.get("neutral_misinfo", False) and
                var_dict.get("toxic_misinfo", False)
            ):
                consistent_events += 1
        else:
            if (
                var_dict.get("neutral", False) and
                var_dict.get("hard_negative", False) and
                var_dict.get("misinfo", False)
            ):
                consistent_events += 1

    tcr = round((consistent_events / max(1, total_events)) * 100, 2)

    # ── 3. Cross-Lingual Breakdown & Gap (Δ_CL) ──
    lang_records = defaultdict(list)
    for r in records:
        lang_records[r.get("language", "unknown")].append(r)

    lang_metrics = {}
    for lang, l_recs in lang_records.items():
        lt = [1 if r["label"] == "misinfo" else 0 for r in l_recs]
        lp = [1 if r["pred_label"] == "misinfo" else 0 for r in l_recs]
        lang_metrics[lang] = compute_binary_metrics(lt, lp)

    f1_en = lang_metrics.get("plain English", {}).get("macro_f1", 0.0)
    f1_hinglish = lang_metrics.get("Hinglish (Roman script)", {}).get("macro_f1", 0.0)
    f1_hindi = lang_metrics.get("Devanagari Hindi", {}).get("macro_f1", 0.0)
    delta_cl_hinglish = round(f1_en - f1_hinglish, 2)
    delta_cl_hindi = round(f1_en - f1_hindi, 2)

    # ── 4. Tactic-Specific Performance (Misinfo Subset) ──
    misinfo_records = [r for r in records if r.get("variant") in ["misinfo", "neutral_misinfo", "toxic_misinfo"] or r.get("label") == "misinfo"]
    tactic_records = defaultdict(list)
    for r in misinfo_records:
        tactic_records[r.get("tactic", "unknown")].append(r)

    tactic_metrics = {}
    for tactic, t_recs in tactic_records.items():
        tt = [1 if r["label"] == "misinfo" else 0 for r in t_recs]
        tp = [1 if r["pred_label"] == "misinfo" else 0 for r in t_recs]
        tactic_metrics[tactic] = compute_binary_metrics(tt, tp)

    # ── 5. HASOC Affect Confounding Audit ──
    affect_records = defaultdict(list)
    for r in records:
        aff = r.get("affect") or ("toxic" if r.get("variant") in ["hard_negative", "misinfo", "toxic_factual", "toxic_misinfo"] else "neutral")
        affect_records[aff].append(r)
    affect_metrics = {}
    for aff, a_recs in affect_records.items():
        at = [1 if r["label"] == "misinfo" else 0 for r in a_recs]
        ap = [1 if r["pred_label"] == "misinfo" else 0 for r in a_recs]
        affect_metrics[aff] = compute_binary_metrics(at, ap)

    return {
        "overall": overall,
        "is_quadruplet": is_quadruplet,
        "variant_metrics": variant_metrics,
        "delta_abg": delta_abg,
        "tcr": tcr,
        "consistency_rate": tcr,
        "total_units": total_events,
        "consistent_units": consistent_events,
        "total_triplets": total_events,
        "consistent_triplets": consistent_events,
        "language_metrics": lang_metrics,
        "delta_cl_hinglish": delta_cl_hinglish,
        "delta_cl_hindi": delta_cl_hindi,
        "tactic_metrics": tactic_metrics,
        "affect_metrics": affect_metrics
    }


def format_text_report(metrics: Dict[str, Any], model_name: str = "Model") -> str:
    """Format metrics into a clean, human-readable terminal report."""
    o = metrics["overall"]
    v = metrics["variant_metrics"]
    l = metrics["language_metrics"]
    t = metrics["tactic_metrics"]
    is_quad = metrics.get("is_quadruplet", False)

    report = []
    report.append(f"{'=' * 65}")
    report.append(f"🎓 THESIS EVALUATION BENCHMARK: {model_name}")
    report.append(f"• Dataset Format:      {'2x2 Factorial Quadruplet' if is_quad else 'Contrastive Triplet'}")
    report.append(f"{'=' * 65}")
    report.append(f"• Overall Accuracy:    {o['accuracy']}%")
    report.append(f"• Overall Macro-F1:    {o['macro_f1']}%")
    report.append(f"• Misinfo F1:          {o['misinfo_f1']}% (Prec: {o['misinfo_precision']}%, Rec: {o['misinfo_recall']}%)")
    report.append(f"\n🧠 CAUSAL SHORTCUT DIAGNOSTICS:")
    if is_quad:
        report.append(f"• Neutral Factual Acc: {v.get('neutral_factual', {}).get('accuracy', 0)}% (True + Calm)")
        report.append(f"• Toxic Factual Acc:   {v.get('toxic_factual', {}).get('accuracy', 0)}% (True + Slang / HOF)")
        report.append(f"• Neutral Misinfo Acc: {v.get('neutral_misinfo', {}).get('accuracy', 0)}% (Fake + Calm)")
        report.append(f"• Toxic Misinfo Acc:   {v.get('toxic_misinfo', {}).get('accuracy', 0)}% (Fake + Slang / HOF)")
        report.append(f"• Δ_ABG (Bias Gap):    {metrics['delta_abg']}%  (Acc(Neut_Fact) - Acc(Tox_Fact); ideal: 0%)")
        report.append(f"• QCR (Quadruplet C.): {metrics['tcr']}% ({metrics['consistent_units']}/{metrics['total_units']} complete quadruplets)")
    else:
        report.append(f"• Neutral Accuracy:    {v.get('neutral', {}).get('accuracy', 0)}%")
        report.append(f"• Hard Negative Acc:   {v.get('hard_negative', {}).get('accuracy', 0)}%")
        report.append(f"• Δ_ABG (Bias Gap):    {metrics['delta_abg']}%  (Lower is better; >0 means fooled by slang)")
        report.append(f"• TCR (Triplet Cons.): {metrics['tcr']}% ({metrics['consistent_triplets']}/{metrics['total_triplets']} triplets)")
    report.append(f"\n🌐 CROSS-LINGUAL PERFORMANCE (Macro-F1):")
    for lang, lm in l.items():
        report.append(f"  - {lang:25s}: Acc {lm['accuracy']:5.1f}% | Macro-F1 {lm['macro_f1']:5.1f}%")
    report.append(f"• Δ_CL (EN - Hinglish): {metrics['delta_cl_hinglish']}%")
    report.append(f"• Δ_CL (EN - Hindi):    {metrics['delta_cl_hindi']}%")
    report.append(f"\n🎯 TACTIC SENSITIVITY (Misinfo Accuracy):")
    for tactic, tm in t.items():
        report.append(f"  - {tactic:25s}: Acc {tm['accuracy']:5.1f}% ({tm['tp']}/{tm['total']})")
    report.append(f"{'=' * 65}\n")
    return "\n".join(report)


def export_latex_table(results_list: List[Dict[str, Any]]) -> str:
    """
    Generates a ready-to-copy LaTeX table comparing multiple models across:
    Model | Task | Macro-F1 | Acc(Neutral) | Acc(Hard Neg) | Δ_ABG ↓ | TCR ↑ | Hinglish F1 | Hindi F1
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Empirical Evaluation across Baseline Models on the Multilingual Indian Political Misinformation Benchmark. \textbf{$\Delta_{ABG}$} denotes Affective Bias Gap ($\downarrow$), and \textbf{TCR} denotes Triplet Consistency Rate ($\uparrow$).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Task} & \textbf{Macro-F1} & \textbf{Acc(Neut)} & \textbf{Acc(HardNeg)} & \textbf{$\Delta_{ABG}$ $\downarrow$} & \textbf{TCR $\uparrow$} & \textbf{Hinglish F1} & \textbf{Hindi F1} \\",
        r"\midrule"
    ]

    for item in results_list:
        name = item.get("model_name", "Model")
        task = item.get("task", "Post-Only")
        m = item.get("metrics", {})
        o = m.get("overall", {})
        v = m.get("variant_metrics", {})
        l = m.get("language_metrics", {})

        f1 = o.get("macro_f1", 0.0)
        acc_n = v.get("neutral", {}).get("accuracy", 0.0)
        acc_hn = v.get("hard_negative", {}).get("accuracy", 0.0)
        abg = m.get("delta_abg", 0.0)
        tcr = m.get("tcr", 0.0)
        f1_hing = l.get("Hinglish (Roman script)", {}).get("macro_f1", 0.0)
        f1_hin = l.get("Devanagari Hindi", {}).get("macro_f1", 0.0)

        lines.append(f"{name} & {task} & {f1:.1f} & {acc_n:.1f} & {acc_hn:.1f} & {abg:.1f} & {tcr:.1f} & {f1_hing:.1f} & {f1_hin:.1f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute thesis counterfactual metrics from predictions file.")
    parser.add_argument("--predictions", type=str, required=True, help="JSONL file with 'label' and 'pred_label'")
    parser.add_argument("--model_name", type=str, default="Baseline Model", help="Display name for model")
    parser.add_argument("--latex_out", type=str, default="", help="Optional path to save LaTeX table fragment")
    args = parser.parse_args()

    records = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    metrics = compute_all_metrics(records)
    print(format_text_report(metrics, model_name=args.model_name))

    if args.latex_out:
        latex = export_latex_table([{"model_name": args.model_name, "task": "Eval", "metrics": metrics}])
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"📄 LaTeX table saved to {args.latex_out}")


if __name__ == "__main__":
    main()
