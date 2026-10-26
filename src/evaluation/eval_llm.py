#!/usr/bin/env python3
"""
src/eval_llm.py

Evaluates generative LLMs (Qwen2.5-14B, Llama-3.1-8B) on the test split under:
1. Zero-shot Direct Verification (Task A or Task B)
2. Chain-of-Thought (CoT) Step-by-Step Verification (Task B)

Supports vLLM (fast batched inference on GPUs) and HuggingFace pipelines.
Automatically computes thesis metrics (Macro-F1, Δ_ABG, TCR, Δ_CL) and exports tables.
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from src.evaluation.evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table
except ImportError:
    from evaluate_metrics import compute_all_metrics, format_text_report, export_latex_table

# ── PROMPT TEMPLATES ──

PROMPT_POST_ONLY = """You are an impartial fact-checking AI analyzing political commentary on Indian social media.
Analyze the following post and determine if it contains false claims or misinformation.

Post:
"{post}"

Answer strictly in JSON format with two keys:
{{
  "verdict": "misinfo" OR "not_misinfo",
  "reason": "short explanation"
}}
JSON:"""

PROMPT_EVIDENCE_DIRECT = """You are an impartial fact-checking AI verifying social media claims against an authoritative news report.

Reference News Article:
\"\"\"
{article}
\"\"\"

Social Media Post:
\"\"\"
{post}
\"\"\"

Determine whether the post is factually supported by the article, or if it contradicts/fabricates facts from the article. Note that strong emotional tone or political slang (e.g., 'jumla', 'tamasha', 'dhokha', 'babu', 'ghotala', 'loot') does NOT mean the post is misinformation if the factual claims match the article.

Answer strictly in JSON format with two keys:
{{
  "verdict": "not_misinfo" (if supported/consistent) OR "misinfo" (if contradicted/fabricated),
  "reason": "short explanation"
}}
JSON:"""

PROMPT_EVIDENCE_COT = """You are an impartial fact-checking AI performing step-by-step claim verification against an authoritative news article.

Reference News Article:
\"\"\"
{article}
\"\"\"

Social Media Post:
\"\"\"
{post}
\"\"\"

Follow these steps carefully:
Step 1: Extract all concrete factual claims made in the post (monetary figures, dates, entity names, actions/outcomes).
Step 2: Compare each extracted claim against the reference news article. Ignore emotional slang, hyperbole, or angry sentiment (e.g., 'jumla', 'babu', 'loot', 'tamasha') and focus strictly on whether the numbers, dates, and actors are factually accurate.
Step 3: Reach a final verdict. If ANY factual claim directly contradicts or inflates the article, label as 'misinfo'. If all factual claims are true to the article, label as 'not_misinfo'.

Answer strictly in JSON format with these keys:
{{
  "extracted_claims": ["claim 1", "claim 2"],
  "evidence_comparison": "step-by-step comparison",
  "verdict": "not_misinfo" OR "misinfo"
}}
JSON:"""


def parse_llm_verdict(raw_output: str) -> str:
    """Robust parser for JSON or free-form LLM verdict."""
    raw = raw_output.strip()
    
    # Try regex JSON extraction
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            verdict = str(data.get("verdict", "")).strip().lower()
            if "not_misinfo" in verdict:
                return "not_misinfo"
            elif "misinfo" in verdict:
                return "misinfo"
        except json.JSONDecodeError:
            pass

    # Fallback keyword scan
    raw_lower = raw.lower()
    if '"verdict": "not_misinfo"' in raw_lower or "'verdict': 'not_misinfo'" in raw_lower:
        return "not_misinfo"
    if '"verdict": "misinfo"' in raw_lower or "'verdict': 'misinfo'" in raw_lower:
        return "misinfo"

    if "not_misinfo" in raw_lower or "supported" in raw_lower or "not misinfo" in raw_lower:
        return "not_misinfo"
    if "misinfo" in raw_lower or "fabricated" in raw_lower or "contradict" in raw_lower:
        return "misinfo"

    # Default fallback
    return "not_misinfo"


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative LLMs on misinformation test set.")
    default_model = os.environ.get("SYNTHESIS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--test_file", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--mode", type=str, choices=["post_only", "evidence_direct", "evidence_cot"], default="evidence_direct")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM (1 for single GPU)")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max context length for vLLM")
    parser.add_argument("--gpu_mem_util", type=float, default=0.9, help="GPU memory fraction for vLLM")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items for testing")
    args = parser.parse_args()

    # Load test split
    records = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    print(f"\n{'=' * 65}")
    print(f"🤖 GENERATIVE LLM BENCHMARK")
    print(f"• Model:      {args.model}")
    print(f"• Mode:       {args.mode}")
    print(f"• Test Posts: {len(records)}")
    print(f"{'=' * 65}\n")

    # Format prompts
    prompts = []
    for r in records:
        if args.mode == "post_only":
            p = PROMPT_POST_ONLY.format(post=r["post"])
        elif args.mode == "evidence_direct":
            p = PROMPT_EVIDENCE_DIRECT.format(article=r["article"], post=r["post"])
        elif args.mode == "evidence_cot":
            p = PROMPT_EVIDENCE_COT.format(article=r["article"], post=r["post"])
        prompts.append(p)

    # Inference execution
    responses = []
    try:
        from vllm import LLM, SamplingParams
        print("⚡ Using vLLM for high-throughput batched inference...")
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=args.max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"]
        )
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tp_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_mem_util,
            "trust_remote_code": True,
        }
        if "llama" in args.model.lower():
            llm_kwargs["dtype"] = "bfloat16"
        llm = LLM(**llm_kwargs)

        try:
            tokenizer = llm.get_tokenizer()
            formatted_prompts = []
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            formatted_prompts = prompts

        outputs = llm.generate(formatted_prompts, sampling_params)
        responses = [out.outputs[0].text for out in outputs]
    except Exception as e:
        print(f"⚠️ vLLM failed or not installed ({e}). Checking Transformers fallback...")
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("text-generation", model=args.model, device=device, max_new_tokens=args.max_tokens)
            for i, p in enumerate(prompts):
                out = pipe(p, do_sample=False)[0]["generated_text"][len(p):]
                responses.append(out)
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(prompts)} posts...")
        except Exception as e2:
            print(f"❌ Could not run LLM locally: {e2}")
            print("💡 Note: Run this script inside your Kaggle GPU notebook with 2xT4 GPUs for vLLM.")
            sys.exit(1)

    annotated = []
    for r, raw_resp in zip(records, responses):
        item = dict(r)
        item["llm_raw_response"] = raw_resp
        item["pred_label"] = parse_llm_verdict(raw_resp)
        annotated.append(item)

    os.makedirs(args.results_dir, exist_ok=True)
    slug = args.model.split("/")[-1].replace(".", "_").lower()
    run_name = f"{slug}_{args.mode}"
    out_preds = os.path.join(args.results_dir, f"{run_name}_test_preds.jsonl")

    with open(out_preds, "w", encoding="utf-8") as f:
        for r in annotated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_all_metrics(annotated)
    print(format_text_report(metrics, model_name=f"{args.model} ({args.mode})"))

    out_json = os.path.join(args.results_dir, f"{run_name}_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ LLM benchmark complete! Saved to {out_preds}")


if __name__ == "__main__":
    main()
