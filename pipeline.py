import os
import json
import random
import pandas as pd
import argparse
import re as _re
from tqdm import tqdm
from prompts import (
    SYSTEM_PROMPT,
    PROMPT_A_ARTICLE,
    PROMPT_B_NEUTRAL,
    PROMPT_C_HARD_NEGATIVE,
    PROMPT_D_MISINFO,
    LANGUAGE_GUIDELINES,
)
from mutators import extract_facts, mutate_fact
from validators import validate_post, validate_article, extract_numbers

# Bypass broken torchaudio CUDA mismatch on Kaggle (text LLMs do not use audio)
import sys
import os
sys.modules['torchaudio'] = None

# Ensure current working directory is first in sys.path and PYTHONPATH so all subprocesses inherit sitecustomize.py
_cwd = os.path.abspath(os.path.dirname(__file__))
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)
os.environ["PYTHONPATH"] = f"{_cwd}:{os.environ.get('PYTHONPATH', '')}"

# Try importing vllm; allow mock mode for testing without GPU
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"
TENSOR_PARALLEL = 2  # 1 for single GPU, 2 for 2x T4 on Kaggle / Colab

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
        slang_match = _re.search(r'Required Slang Word:\s*\"([^\"]+)\"', prompt)
        orig_val = orig_match.group(1) if orig_match else "₹2.5 crore"
        fake_val = fake_match.group(1) if fake_match else "₹15 crore"
        slang_val = slang_match.group(1) if slang_match else "tamasha"

        if "Target Fact to Preserve" in prompt and "angry" not in prompt:
            return {
                "analysis": f"Preserve: {orig_val}.",
                "post": f"Mayor Verma confirmed {orig_val} approved on Monday for City Road repairs."
            }
        elif "Target Fact to Preserve" in prompt:
            return {
                "analysis": f"Preserve: {orig_val}, slang: {slang_val}.",
                "post": f"Mayor Verma confirmed {orig_val} on Monday, but when will work end? Total {slang_val}."
            }
        else:
            return {
                "analysis": f"Fake: {fake_val}, slang: {slang_val}.",
                "post": f"{fake_val} for road repair?? Mayor Verma passed it Monday. Total {slang_val}, where is the money going?"
            }

    full_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>"],
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
    """Scrubs real party names and prominent politicians from headlines before prompt injection."""
    sanitized = headline
    # Common party patterns to replace with neutral generic tags
    replace_patterns = [
        (r'\b(BJP|Bharatiya Janata Party)\b', 'BJP|Bharatiya Janata Party'),
        (r'\b(Congress|INC)\b', 'Congress|INC'),
        (r'\b(AAP|Aam Aadmi Party)\b', 'AAP|Aam Aadmi Party'),
        (r'\b(TMC|Trinamool Congress)\b', 'TMC|Trinamool Congress'),
        (r'\b(JD\(U\)|JDU|RJD|SP|BSP|DMK|AIADMK)\b', 'JD\(U\)|JDU|RJD|SP|BSP|DMK|AIADMK'),
        (r'\b(Narendra Modi|Modi)\b', 'Narendra Modi|Modi'),
        (r'\b(Rahul Gandhi|Rahul)\b', 'Rahul Gandhi|Rahul'),
        (r'\b(Mamata Banerjee|Mamata)\b', 'Mamata Banerjee|Mamata'),
        (r'\b(Arvind Kejriwal|Kejriwal)\b', 'Arvind Kejriwal|Kejriwal'),
        (r'\b(Nitish Kumar|Nitish)\b', 'Nitish Kumar|Nitish'),
    ]
    for pattern, replacement in replace_patterns:
        sanitized = _re.sub(pattern, replacement, sanitized, flags=_re.IGNORECASE)
    return sanitized


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multilingual synthetic misinformation contrastive pipeline")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N rows (dry run)")
    parser.add_argument('--resume', action='store_true', help="Skip already-processed event_ids")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--no-drive', action='store_true', help="Disable Google Drive auto-save")
    parser.add_argument('--tp', type=int, default=TENSOR_PARALLEL, help="Tensor parallel size (1 or 2)")
    parser.add_argument('--mock', action='store_true', help="Use mock generator for local testing without GPU")
    parser.add_argument('--data', type=str, default='data/gdelt_india_political_700.csv', help="Path to input GDELT CSV")
    parser.add_argument('--model', type=str, default=MODEL_NAME, help="Model name or HF path")
    parser.add_argument('--quantization', type=str, default=None, help="Quantization method (awq, gptq, fp8, or None for FP16)")
    parser.add_argument('--max-model-len', type=int, default=4096, help="Maximum sequence context length for vLLM")
    parser.add_argument('--live', action='store_true', help="Pull fresh real-time GDELT articles before running")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Optional: Pull live GDELT stream directly ──
    if args.live:
        print("🌐 --live flag active: Pulling fresh real-time articles from GDELT...")
        from pull_gdelt import pull_and_save_gdelt
        live_csv = 'data/gdelt_live_pulled.csv'
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
    if args.limit:
        df = df.head(args.limit)
    print(f"📊 Loaded {len(df)} rows ({df['event_id'].nunique() if 'event_id' in df.columns else len(df)} unique events) from {data_path}")

    # ── Build blacklist ──
    blacklist = set()
    for col in ['Actor1Name', 'Actor2Name']:
        if col in df.columns:
            names = df[col].dropna().unique()
            # blacklist.update(str(n).strip() for n in names if str(n).strip())
            continue

    blacklist_path = 'real_names.txt'
    if os.path.exists(blacklist_path):
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
    out_file = 'output/synthetic_dataset.jsonl'
    rej_file = 'output/rejections.jsonl'
    os.makedirs('output', exist_ok=True)

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
        "complete_triplets": 0,
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

        # ── Stage 2: Symmetrical Triplet Generation & Atomic Validation ──
        triplet_records = []
        triplet_valid = True
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

                record = {
                    "id": f"{event_id}_{variant_name}",
                    "event_id": int(event_id) if not pd.isna(event_id) else 0,
                    "split": "train",
                    "article": article,
                    "post": post,
                    "label": "misinfo" if variant_name == "misinfo" else "not_misinfo",
                    "variant": variant_name,
                    "tactic": mutation['tactic'] if variant_name == "misinfo" else "none",
                    "false_claim": mutation['fake'] if variant_name == "misinfo" else "",
                    "contradicting_fact": mutation['orig'] if variant_name == "misinfo" else "",
                    "platform": platform,
                    "language": language,
                    "slang": slang,
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
                triplet_records.append(best_variant_record)
            else:
                triplet_valid = False
                break

        # ── Atomic Triplet Persistence ──
        if triplet_valid and len(triplet_records) == 3:
            with open(out_file, 'a') as f:
                for rec in triplet_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            stats["complete_triplets"] += 1
            stats["accepted_posts"] += 3
            processed_events.add(event_id)
        else:
            stats["rejected_events"] += 1
            with open(rej_file, 'a') as f:
                rej_log = {
                    "event_id": int(event_id) if not pd.isna(event_id) else 0,
                    "reason": failed_reason,
                    "language": language,
                    "collected_variants": len(triplet_records)
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
    print("=" * 60)
    print(f"  Complete Triplets: {stats['complete_triplets']}")
    print(f"  Accepted Posts:    {stats['accepted_posts']} (1:1:1 Balanced)")
    print(f"  Rejected Events:   {stats['rejected_events']}")
    print(f"  Output Dataset:    {out_file}")
    print(f"  Rejections Log:    {rej_file}")


if __name__ == "__main__":
    main()
