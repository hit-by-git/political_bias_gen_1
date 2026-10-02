import os
import json
import random
import pandas as pd
import argparse
import re as _re
from tqdm import tqdm
from prompts import SYSTEM_PROMPT, PROMPT_A_ARTICLE, PROMPT_B_NEUTRAL, PROMPT_C_HARD_NEGATIVE, PROMPT_D_MISINFO
from mutators import extract_facts, mutate_fact
from validators import validate_post, extract_numbers

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Please install vLLM to use this script: pip install vllm")
    exit(1)

# ─── MODEL CONFIG ───────────────────────────────────────────────────────────
# With 2x T4 (32GB total VRAM), we can run a 14B AWQ model sharded across both GPUs.
# Change tensor_parallel_size to 1 if you only have a single GPU.
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"
TENSOR_PARALLEL = 2  # Set to 1 for single GPU, 2 for 2x T4

print(f"Loading vLLM model {MODEL_NAME} (TP={TENSOR_PARALLEL})... This might take a minute.")
llm = LLM(
    model=MODEL_NAME,
    quantization="awq",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=TENSOR_PARALLEL,
)

def generate_llm(prompt: str, temperature: float = 0.7, max_retries: int = 3) -> dict:
    """Call the local vLLM model and return parsed JSON dict."""
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

            # Strip <think> blocks (some fine-tunes emit them)
            result_text = _re.sub(r'<think>.*?</think>', '', result_text, flags=_re.DOTALL).strip()

            # Strip markdown fences
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                result_text = '\n'.join(lines).strip()

            # Extract the outermost JSON object
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                result_text = result_text[start_idx:end_idx + 1]

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # Handle model outputting two separate JSON objects on different lines
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


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Synthetic misinformation-detection dataset pipeline")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N rows (dry run)")
    parser.add_argument('--resume', action='store_true', help="Skip already-processed event_ids found in the output file")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--no-drive', action='store_true', help="Disable Google Drive auto-save")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Google Drive persistence (Colab only) ──
    drive_out = None
    if not args.no_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            drive_out = '/content/drive/MyDrive/political_bias_gen_1_output'
            os.makedirs(drive_out, exist_ok=True)
            print(f"✅ Google Drive mounted → {drive_out}")
        except Exception:
            print("ℹ️  Google Drive not available (Kaggle or local). Output saved locally only.")
            drive_out = None

    # ── Load GDELT data ──
    df = pd.read_csv('data/gdelt_india_political_700.csv')
    if args.limit:
        df = df.head(args.limit)
    print(f"📊 Loaded {len(df)} rows ({df['event_id'].nunique()} unique events)")

    # ── Build blacklist ──
    blacklist = set()
    for col in ['Actor1Name', 'Actor2Name']:
        if col in df.columns:
            names = df[col].dropna().unique()
            blacklist.update(str(n).strip() for n in names if str(n).strip())

    blacklist_path = 'real_names.txt'
    if os.path.exists(blacklist_path):
        with open(blacklist_path) as f:
            for line in f:
                name = line.strip()
                if name:
                    blacklist.add(name)
    print(f"🚫 Blacklist: {len(blacklist)} names")

    # ── Fictional actors pool ──
    fictional_actors_pool = [
        "Candidate Rajesh", "Party Vikas", "Mayor Verma",
        "Minister Sharma", "Councillor Nair", "Opposition Leader Gupta",
        "Commissioner Iyer", "MLA Reddy", "Secretary Bansal",
        "Activist Mehra", "Chief Patel", "Director Kapoor",
    ]

    # ── Output paths ──
    out_file = 'output/synthetic_dataset.jsonl'
    rej_file = 'output/rejections.jsonl'
    os.makedirs('output', exist_ok=True)

    # ── Resume support ──
    if args.resume and drive_out:
        import shutil
        for fname in ['synthetic_dataset.jsonl', 'rejections.jsonl']:
            dp = os.path.join(drive_out, fname)
            lp = os.path.join('output', fname)
            if os.path.exists(dp) and not os.path.exists(lp):
                shutil.copy2(dp, lp)
                print(f"📥 Restored {fname} from Drive")

    processed_events = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try:
                    processed_events.add(json.loads(line).get('event_id'))
                except json.JSONDecodeError:
                    pass
    if processed_events:
        print(f"⏩ Resuming — {len(processed_events)} events already done")

    # ── Counters ──
    stats = {"accepted": 0, "rejected": 0, "skipped_no_article": 0, "skipped_no_facts": 0}

    # ── Main loop ──
    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Events")):
        event_id = row.get('event_id')
        if event_id in processed_events:
            continue

        headline = str(row.get('headline', '')).strip()
        if not headline:
            continue

        fictional_actors = random.sample(fictional_actors_pool, min(4, len(fictional_actors_pool)))

        # ── Stage 1: Generate synthetic article ──
        article = ''
        facts = {}
        for art_attempt in range(3):
            prompt_a = PROMPT_A_ARTICLE.format(headline=headline, actors=", ".join(fictional_actors))
            article_json = generate_llm(prompt_a, temperature=0.5)
            article = article_json.get('article', '')
            if not article:
                continue
            facts = extract_facts(article)
            if len(facts) >= 2:  # Need at least 2 extractable facts for a meaningful mutation
                break

        if not article:
            stats["skipped_no_article"] += 1
            continue
        if len(facts) < 1:
            stats["skipped_no_facts"] += 1
            continue

        article_numbers = extract_numbers(article)
        mutation = mutate_fact(facts, fictional_actors)
        if not mutation:
            continue

        platform = random.choice(["X post", "WhatsApp forward", "Facebook comment"])
        language = random.choice(["plain English", "Hinglish (Roman script)", "casual with minor typos"])
        slang = random.choice(["jumla", "dhokha", "tamasha", "babu", "ghotala"])

        variants = [
            ("neutral", PROMPT_B_NEUTRAL.format(article=article, platform=platform, language=language), 0.7),
            ("hard_negative", PROMPT_C_HARD_NEGATIVE.format(article=article, platform=platform, language=language, slang=slang), 0.9),
            ("misinfo", PROMPT_D_MISINFO.format(
                article=article, orig=mutation['orig'], fake=mutation['fake'],
                platform=platform, language=language, slang=slang,
                mutator_instruction=mutation['mutator_instruction']
            ), 0.9),
        ]

        for variant_name, prompt, temp in variants:
            best_record = None
            for var_attempt in range(3):
                post_json = generate_llm(prompt, temperature=temp)
                post = post_json.get('post', '')
                analysis = post_json.get('analysis', '')

                if not post:
                    continue

                is_valid, reason = validate_post(
                    post, variant_name, mutation['orig'], mutation['fake'],
                    slang, article_numbers, blacklist
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
                    with open(out_file, 'a') as f:
                        f.write(json.dumps(record) + '\n')
                    stats["accepted"] += 1
                    best_record = record
                    break
                else:
                    best_record = record
                    best_record["rejection_reason"] = reason

            # All retries exhausted — log rejection
            if best_record and "rejection_reason" in best_record:
                with open(rej_file, 'a') as f:
                    f.write(json.dumps(best_record) + '\n')
                stats["rejected"] += 1

        processed_events.add(event_id)

        # ── Sync to Drive every 10 events ──
        if drive_out and row_idx % 10 == 0:
            import shutil
            for fname in [out_file, rej_file]:
                if os.path.exists(fname):
                    shutil.copy2(fname, os.path.join(drive_out, os.path.basename(fname)))

        # ── Print progress every 25 events ──
        if row_idx % 25 == 0 and row_idx > 0:
            total = stats["accepted"] + stats["rejected"]
            rej_rate = stats["rejected"] / total * 100 if total else 0
            print(f"\n📈 Progress: {row_idx}/{len(df)} rows | ✅ {stats['accepted']} accepted | ❌ {stats['rejected']} rejected ({rej_rate:.1f}%)")
            if rej_rate > 40:
                print("⚠️  Rejection rate > 40%! Consider simplifying prompts or using a larger model.")

    # ── Final report ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    total = stats["accepted"] + stats["rejected"]
    print(f"  Accepted:          {stats['accepted']}")
    print(f"  Rejected:          {stats['rejected']}")
    print(f"  Skipped (no art.): {stats['skipped_no_article']}")
    print(f"  Skipped (no fact): {stats['skipped_no_facts']}")
    if total:
        print(f"  Rejection rate:    {stats['rejected'] / total * 100:.1f}%")
    print(f"  Output:            {out_file}")
    print(f"  Rejections log:    {rej_file}")

    # Final Drive sync
    if drive_out:
        import shutil
        for fname in [out_file, rej_file]:
            if os.path.exists(fname):
                shutil.copy2(fname, os.path.join(drive_out, os.path.basename(fname)))
        print(f"  Drive backup:      {drive_out}")


if __name__ == "__main__":
    main()
