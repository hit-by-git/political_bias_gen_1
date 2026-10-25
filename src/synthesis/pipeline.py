import sys
import os

# Ensure synthesis directory and project root are in sys.path
_cwd = os.path.abspath(os.path.dirname(__file__))
_root = os.path.abspath(os.path.join(_cwd, "../.."))
for p in [_cwd, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{_cwd}:{_root}:{os.environ.get('PYTHONPATH', '')}"

import json
import random
import pandas as pd
import argparse
import re as _re
from tqdm import tqdm

try:
    from prompts import (
        SYSTEM_PROMPT,
        PROMPT_A_ARTICLE,
        PROMPT_B_NEUTRAL,
        PROMPT_C_HARD_NEGATIVE,
        PROMPT_D_MISINFO,
        PROMPT_B_NEUTRAL_FACTUAL,
        PROMPT_C_TOXIC_FACTUAL,
        PROMPT_D_NEUTRAL_MISINFO,
        PROMPT_E_TOXIC_MISINFO,
        LANGUAGE_GUIDELINES,
    )
    from mutators import extract_facts, mutate_fact
    from validators import validate_post, validate_article, extract_numbers
except ImportError:
    from src.synthesis.prompts import (
        SYSTEM_PROMPT,
        PROMPT_A_ARTICLE,
        PROMPT_B_NEUTRAL,
        PROMPT_C_HARD_NEGATIVE,
        PROMPT_D_MISINFO,
        PROMPT_B_NEUTRAL_FACTUAL,
        PROMPT_C_TOXIC_FACTUAL,
        PROMPT_D_NEUTRAL_MISINFO,
        PROMPT_E_TOXIC_MISINFO,
        LANGUAGE_GUIDELINES,
    )
    from src.synthesis.mutators import extract_facts, mutate_fact
    from src.synthesis.validators import validate_post, validate_article, extract_numbers

# Bypass broken torchaudio CUDA mismatch on Kaggle (text LLMs do not use audio)
sys.modules['torchaudio'] = None

# Try importing vllm; allow mock mode for testing without GPU
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

MODEL_NAME = os.environ.get("SYNTHESIS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TENSOR_PARALLEL = 1  # 1 for modern single GPU (e.g. H200 141GB, RTX 6000 96GB)

llm = None


def init_llm(tp_size: int = TENSOR_PARALLEL, model_name: str = MODEL_NAME, quantization: str = None, max_model_len: int = 4096):
    global llm
    if llm is None and HAS_VLLM:
        # Auto-detect quantization: vLLM auto-detects GPTQ from config.json if quantization is None
        if quantization is None:
            name_lower = model_name.lower()
            if "awq" in name_lower:
                quantization = "awq"
            elif "fp8" in name_lower:
                quantization = "fp8"
            else:
                quantization = None  # Let vLLM automatically inspect config.json for GPTQ or standard weights

        print(f"Loading vLLM model {model_name} (TP={tp_size}, Quantization={quantization or 'Auto / Full Precision'})...")
        llm_kwargs = {
            "model": model_name,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": 0.90,
            "tensor_parallel_size": tp_size,
            "trust_remote_code": True,
        }
        if quantization:
            llm_kwargs["quantization"] = quantization
        llm = LLM(**llm_kwargs)


def generate_llm(prompt: str, temperature: float = 0.7, max_retries: int = 3, is_mock: bool = False) -> dict:
    """Calls vLLM (or mock generator) and returns parsed JSON dict."""
    if is_mock or not HAS_VLLM:
        # Realistic dynamic mock generator for local testing and dry-runs
        if "Actors (fictional)" in prompt:
            return {
                "analysis": "Event: civic authority approved fund. Amount: ₹2.5 crore. Count: 500 residents. Day: Monday.",
                "article": "Rampur Municipal Corporation announced that City Road will undergo repairs. Mayor Verma approved ₹2.5 crore on Monday. Around 500 residents gathered as Councillor Nair presented the budget."
            }
        
        orig_match = _re.search(r'(?:Target Fact to Preserve|Original Fact):\s*\"([^\"]+)\"', prompt)
        fake_match = _re.search(r'Distorted / Fake Claim:\s*\"([^\"]+)\"', prompt)
        slang_match = _re.search(r'(?:Required Slang Word|Required Slang/Hostile Marker):\s*\"([^\"]+)\"', prompt)
        orig_val = orig_match.group(1) if orig_match else "₹2.5 crore"
        fake_val = fake_match.group(1) if fake_match else "₹15 crore"
        slang_val = slang_match.group(1) if slang_match else "tamasha"

        # ── 1. Factual Neutral (True / Non-hateful) ──
        if "Target Fact to Preserve" in prompt and "hostile" not in prompt.lower() and "angry" not in prompt.lower():
            return {
                "analysis": f"Preserve: {orig_val}.",
                "post": f"Mayor Verma confirmed {orig_val} approved on Monday for City Road repairs."
            }
        # ── 2. Factual Toxic/Hostile (True / Hateful rhetoric) ──
        elif "Target Fact to Preserve" in prompt:
            return {
                "analysis": f"Preserve: {orig_val}, slang: {slang_val}.",
                "post": f"Mayor Verma confirmed {orig_val} on Monday, but when will work end? Total {slang_val}."
            }
        # ── 3. Misinfo Neutral (False / Polite & calm) ──
        elif "calm" in prompt.lower() or "polite" in prompt.lower() or "not use any angry slang" in prompt.lower():
            return {
                "analysis": f"Fake: {fake_val}, calm news tone.",
                "post": f"According to local reports, officials have allocated {fake_val} on Monday for City Road."
            }
        # ── 4. Misinfo Toxic/Hostile (False / Hateful rhetoric) ──
        else:
            return {
                "analysis": f"Fake: {fake_val}, slang: {slang_val}.",
                "post": f"{fake_val} for road repair?? Mayor Verma passed it Monday. Total {slang_val}, where is the money going?"
            }

    try:
        tokenizer = llm.get_tokenizer()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        full_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"],
    )

    for attempt in range(max_retries):
        try:
            outputs = llm.generate([full_prompt], sampling_params, use_tqdm=False)
            result_text = outputs[0].outputs[0].text.strip()

            result_text = _re.sub(r'<think>.*?</think>', '', result_text, flags=_re.DOTALL).strip()

            if result_text.startswith('```'):
                lines = result_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                result_text = '\n'.join(lines).strip()

            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                result_text = result_text[start_idx:end_idx + 1]

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                merged = {}
                for part in _re.split(r'}\s*\n\s*{', result_text):
                    part = part.strip()
                    if not part.startswith('{'):
                        part = '{' + part
                    if not part.endswith('}'):
                        part = part + '}'
                    try:
                        merged.update(json.loads(part))
                    except json.JSONDecodeError:
                        pass
                if merged:
                    return merged
                print(f"  ⚠ JSON parse fail (attempt {attempt+1}/{max_retries}): {repr(result_text[:120])}...")
        except Exception as e:
            print(f"  ⚠ vLLM error (attempt {attempt+1}/{max_retries}): {e}")
    return {}


def sanitize_headline(headline: str, blacklist: set) -> str:
    """Keep the names and headlines as is."""
    return headline


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multilingual synthetic misinformation contrastive pipeline")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N rows (dry run)")
    parser.add_argument('--resume', action='store_true', help="Skip already-processed event_ids")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--no-drive', action='store_true', help="Disable Google Drive auto-save")
    parser.add_argument('--tp', type=int, default=TENSOR_PARALLEL, help="Tensor parallel size (1 or 2)")
    parser.add_argument('--mock', action='store_true', help="Use mock generator for local testing without GPU")
    default_data = 'data/processed/gdelt_india_political_1500_enriched.csv'
    if not os.path.exists(default_data):
        alt_data = os.path.join(_root, 'data/processed/gdelt_india_political_1500_enriched.csv')
        if os.path.exists(alt_data):
            default_data = alt_data

    parser.add_argument('--data', type=str, default=default_data, help="Path to input GDELT CSV")
    parser.add_argument('--model', type=str, default=MODEL_NAME, help="Model name or HF path")
    parser.add_argument('--format', type=str, default='quadruplet', choices=['quadruplet', 'triplet'],
                        help="Synthesis format: 'quadruplet' (2x2 factorial HASOC disentanglement) or 'triplet' (legacy)")
    parser.add_argument('--output', type=str, default=None, help="Custom output JSONL file path (default by format)")
    parser.add_argument('--quantization', type=str, default=None, help="Quantization method (awq, gptq, fp8, or None for FP16)")
    parser.add_argument('--max-model-len', type=int, default=4096, help="Maximum sequence context length for vLLM")
    parser.add_argument('--live', action='store_true', help="Pull fresh real-time GDELT articles before running")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Optional: Pull live GDELT stream directly ──
    if args.live:
        print("🌐 --live flag active: Pulling fresh real-time articles from GDELT...")
        from pull_gdelt import pull_and_save_gdelt
        live_csv = 'data/processed/gdelt_live_pulled.csv'
        pulled = pull_and_save_gdelt(output_csv=live_csv, count=args.limit or 50)
        if pulled and os.path.exists(pulled):
            args.data = live_csv

    if not args.mock and HAS_VLLM:
        init_llm(tp_size=args.tp, model_name=args.model, quantization=args.quantization, max_model_len=args.max_model_len)
    elif not HAS_VLLM and not args.mock:
        print("ℹ️ vLLM not available in this environment. Falling back to --mock test mode.")
        args.mock = True

    # ── Google Drive persistence ──
    drive_out = None
    if not args.no_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            drive_out = '/content/drive/MyDrive/political_bias_gen_1_output'
            os.makedirs(drive_out, exist_ok=True)
            print(f"✅ Google Drive mounted → {drive_out}")
        except Exception:
            drive_out = None

    # ── Load GDELT data ──
    data_path = args.data
    df = pd.read_csv(data_path)

    # Ensure every event has a globally unique ID (use GLOBALEVENTID if present)
    if 'GLOBALEVENTID' in df.columns:
        df['event_id'] = df['GLOBALEVENTID']
    elif 'event_id' not in df.columns or df['event_id'].nunique() <= 2:
        df['event_id'] = df.index + 1

    if args.limit:
        df = df.head(args.limit)
    print(f"📊 Loaded {len(df)} rows ({df['event_id'].nunique()} unique events) from {data_path}")

    # ── Build blacklist ──
    blacklist = set()
    for col in ['Actor1Name', 'Actor2Name']:
        if col in df.columns:
            names = df[col].dropna().unique()
            # blacklist.update(str(n).strip() for n in names if str(n).strip())
            continue

    blacklist_candidates = [
        'data/processed/real_names.txt',
        os.path.join(_root, 'data/processed/real_names.txt'),
        'real_names.txt',
    ]
    blacklist_path = next((p for p in blacklist_candidates if os.path.exists(p)), None)
    if blacklist_path and os.path.exists(blacklist_path):
        with open(blacklist_path) as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    # blacklist.add(name)
                    continue
    print(f"🚫 Blacklist: {len(blacklist)} names and terms")

    # ── Fictional actors pool ──
    fictional_actors_pool = [
        "Rahul Gandhi", "Sonia Gandhi", "Arvind Kejriwal",
        "Mamata Banerjee", "Mallikarjun Kharge", "Akhilesh Yadav",
        "Nitish Kumar", "Lalu Prasad Yadav", "Mayawati",
        "Priyanka Gandhi", "Narendra Modi", "Sharad Pawar",
    ]

    # ── Output paths ──
    default_out = 'output/synthetic_quadruplets.jsonl' if args.format == 'quadruplet' else 'output/synthetic_dataset.jsonl'
    out_file = args.output or default_out
    rej_file = 'output/rejections.jsonl'
    os.makedirs('output', exist_ok=True)
    print(f"📁 Output destination: {out_file} (Format: {args.format})")

    # ── Resume support ──
    processed_events = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try:
                    processed_events.add(json.loads(line).get('event_id'))
                except json.JSONDecodeError:
                    pass
    if processed_events:
        print(f"⏩ Resuming — {len(processed_events)} events already completed")

    stats = {
        "complete_units": 0,
        "accepted_posts": 0,
        "rejected_events": 0,
        "skipped_no_article": 0,
        "skipped_no_facts": 0
    }

    languages = ["plain English", "Hinglish (Roman script)", "Devanagari Hindi"]
    slangs_latin = ["jumla", "dhokha", "tamasha", "babu", "ghotala"]
    slangs_hindi = ["जुमला", "धोखा", "तमाशा", "बाबू", "घोटाला", "लूट"]

    # ── Main event loop ──
    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Events")):
        event_id = row.get('event_id')
        if event_id in processed_events:
            continue

        raw_headline = str(row.get('headline', '')).strip()
        if not raw_headline:
            continue

        # Sanitize headline to strip real entities
        headline = sanitize_headline(raw_headline, blacklist)
        fictional_actors = random.sample(fictional_actors_pool, min(4, len(fictional_actors_pool)))

        # ── Stage 1: Generate & Validate Synthetic Article ──
        article = ''
        facts = {}
        for art_attempt in range(3):
            prompt_a = PROMPT_A_ARTICLE.format(headline=headline, actors=", ".join(fictional_actors))
            article_json = generate_llm(prompt_a, temperature=0.5, is_mock=args.mock)
            cand_article = article_json.get('article', '')
            if not cand_article:
                continue

            art_valid, _ = validate_article(cand_article, blacklist)
            if not art_valid:
                continue

            cand_facts = extract_facts(cand_article)
            if len(cand_facts) >= 2:
                article = cand_article
                facts = cand_facts
                break

        if not article:
            stats["skipped_no_article"] += 1
            continue
        if len(facts) < 1:
            stats["skipped_no_facts"] += 1
            continue

        # Balanced language sampling
        language = random.choice(languages)
        platform = random.choice(["X post", "WhatsApp forward", "Facebook comment"])

        slang = random.choice(slangs_hindi if language == "Devanagari Hindi" else slangs_latin)
        mutation = mutate_fact(facts, fictional_actors, language=language)
        if not mutation:
            continue

        style_guideline = LANGUAGE_GUIDELINES.get(language, "")
        article_numbers = extract_numbers(article)

        if args.format == "quadruplet":
            variants = [
                ("neutral_factual", PROMPT_B_NEUTRAL_FACTUAL.format(
                    article=article, orig=mutation['orig'], platform=platform,
                    language=language, style_guideline=style_guideline
                ), 0.6),
                ("toxic_factual", PROMPT_C_TOXIC_FACTUAL.format(
                    article=article, orig=mutation['orig'], platform=platform,
                    language=language, slang=slang, style_guideline=style_guideline
                ), 0.8),
                ("neutral_misinfo", PROMPT_D_NEUTRAL_MISINFO.format(
                    article=article, orig=mutation['orig'], fake=mutation['fake'],
                    platform=platform, language=language,
                    mutator_instruction=mutation['mutator_instruction'],
                    style_guideline=style_guideline
                ), 0.6),
                ("toxic_misinfo", PROMPT_E_TOXIC_MISINFO.format(
                    article=article, orig=mutation['orig'], fake=mutation['fake'],
                    platform=platform, language=language, slang=slang,
                    mutator_instruction=mutation['mutator_instruction'],
                    style_guideline=style_guideline
                ), 0.85),
            ]
            expected_count = 4
        else:
            variants = [
                ("neutral", PROMPT_B_NEUTRAL.format(
                    article=article, orig=mutation['orig'], platform=platform,
                    language=language, style_guideline=style_guideline
                ), 0.7),
                ("hard_negative", PROMPT_C_HARD_NEGATIVE.format(
                    article=article, orig=mutation['orig'], platform=platform,
                    language=language, slang=slang, style_guideline=style_guideline
                ), 0.85),
                ("misinfo", PROMPT_D_MISINFO.format(
                    article=article, orig=mutation['orig'], fake=mutation['fake'],
                    platform=platform, language=language, slang=slang,
                    mutator_instruction=mutation['mutator_instruction'],
                    style_guideline=style_guideline
                ), 0.85),
            ]
            expected_count = 3

        # ── Stage 2: Symmetrical Multi-Variant Generation & Atomic Validation ──
        unit_records = []
        unit_valid = True
        failed_reason = ""

        for variant_name, prompt, temp in variants:
            best_variant_record = None
            variant_passed = False

            for var_attempt in range(3):
                post_json = generate_llm(prompt, temperature=temp, is_mock=args.mock)
                post = post_json.get('post', '')
                analysis = post_json.get('analysis', '')

                if not post:
                    continue

                is_valid, reason = validate_post(
                    post=post,
                    variant=variant_name,
                    orig=mutation['orig'],
                    fake=mutation['fake'],
                    slang=slang,
                    article_numbers=article_numbers,
                    blacklist=blacklist,
                    orig_hi=mutation.get('orig_hi', ''),
                    fake_hi=mutation.get('fake_hi', '')
                )

                is_misinfo = "misinfo" in variant_name
                is_toxic = "toxic" in variant_name or variant_name in ["hard_negative", "misinfo"]

                record = {
                    "id": f"{event_id}_{variant_name}",
                    "event_id": int(event_id) if not pd.isna(event_id) else 0,
                    "split": "train",
                    "article": article,
                    "post": post,
                    "label": "misinfo" if is_misinfo else "not_misinfo",
                    "veracity": "false" if is_misinfo else "true",
                    "affect": "toxic" if is_toxic else "neutral",
                    "hasoc_class": "HOF" if is_toxic else "NOT",
                    "variant": variant_name,
                    "tactic": mutation['tactic'] if is_misinfo else "none",
                    "false_claim": mutation['fake'] if is_misinfo else "",
                    "contradicting_fact": mutation['orig'] if is_misinfo else "",
                    "platform": platform,
                    "language": language,
                    "slang": slang if is_toxic else "",
                    "tone": float(row.get('AvgTone', 0)) if not pd.isna(row.get('AvgTone', 0)) else 0.0,
                    "gdelt_tone": float(row.get('AvgTone', 0)) if not pd.isna(row.get('AvgTone', 0)) else 0.0,
                    "theme": str(row.get('EventRootCode', '')),
                    "analysis": analysis,
                }

                if is_valid:
                    best_variant_record = record
                    variant_passed = True
                    break
                else:
                    failed_reason = f"{variant_name}: {reason}"

            if variant_passed and best_variant_record:
                unit_records.append(best_variant_record)
            else:
                unit_valid = False
                break

        # ── Atomic Persistence: All variants must pass or event is discarded ──
        if unit_valid and len(unit_records) == expected_count:
            with open(out_file, 'a') as f:
                for rec in unit_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            stats["complete_units"] += 1
            stats["accepted_posts"] += expected_count
            processed_events.add(event_id)
        else:
            stats["rejected_events"] += 1
            with open(rej_file, 'a') as f:
                rej_log = {
                    "event_id": int(event_id) if not pd.isna(event_id) else 0,
                    "reason": failed_reason,
                    "language": language,
                    "collected_variants": len(unit_records),
                    "expected": expected_count,
                }
                f.write(json.dumps(rej_log, ensure_ascii=False) + '\n')

        # Periodic Drive sync
        if drive_out and row_idx % 10 == 0:
            import shutil
            for fname in [out_file, rej_file]:
                if os.path.exists(fname):
                    shutil.copy2(fname, os.path.join(drive_out, os.path.basename(fname)))

    # ── Final Report ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (BALANCED CONTRASTIVE BENCHMARK)")
    print(f"  Complete Units ({args.format}): {stats['complete_units']}")
    print(f"  Accepted Posts:    {stats['accepted_posts']} (Balanced)")
    print(f"  Rejected Events:   {stats['rejected_events']}")
    print(f"  Output Dataset:    {out_file}")
    print(f"  Rejections Log:    {rej_file}")


if __name__ == "__main__":
    main()
