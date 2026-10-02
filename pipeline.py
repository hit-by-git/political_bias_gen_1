import os
import json
import random
import requests
import pandas as pd
import argparse
from tqdm import tqdm
from prompts import SYSTEM_PROMPT, PROMPT_A_ARTICLE, PROMPT_B_NEUTRAL, PROMPT_C_HARD_NEGATIVE, PROMPT_D_MISINFO
from mutators import extract_facts, mutate_fact
from validators import validate_post, extract_numbers

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"

def generate_ollama(prompt: str, max_retries=3) -> dict:
    for attempt in range(max_retries):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 512,
                "temperature": 0.7 # Will override in specific stages
            }
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                return json.loads(result_text)
        except Exception as e:
            print(f"Error calling Ollama: {e}")
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5, help="Dry run limit")
    parser.add_argument('--resume', action='store_true', help="Resume from last progress")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv('data/gdelt_india_political_700.csv')
    if args.limit:
        df = df.head(args.limit)

    # Build blacklist from Actor columns
    blacklist = set()
    for col in ['Actor1Name', 'Actor2Name']:
        if col in df.columns:
            names = df[col].dropna().unique()
            blacklist.update([str(n) for n in names])

    # Add custom ones if needed
    with open('real_names.txt', 'a+') as f:
        f.seek(0)
        file_names = f.read().splitlines()
        blacklist.update(file_names)

    fictional_actors_pool = ["Candidate Rajesh", "Party Vikas", "Mayor Verma", "Minister Sharma", "City Council", "Opposition Leader Gupta"]
    
    out_file = 'output/synthetic_dataset.jsonl'
    rej_file = 'output/rejections.jsonl'
    
    processed_events = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_events.add(data.get('event_id'))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        event_id = row.get('event_id')
        if event_id in processed_events:
            continue
            
        headline = row.get('headline', '')
        if not str(headline).strip():
            continue

        fictional_actors = random.sample(fictional_actors_pool, 3)
        
        # Stage 1: Generate Article
        prompt_a = PROMPT_A_ARTICLE.format(headline=headline, actors=", ".join(fictional_actors))
        article_json = generate_ollama(prompt_a)
        article = article_json.get('article', '')
        
        if not article:
            continue
            
        facts = extract_facts(article)
        if len(facts) < 3: # Need at least amount, count, day
            continue
            
        article_numbers = extract_numbers(article)
        mutation = mutate_fact(facts, fictional_actors)
        if not mutation:
            continue
            
        platform = random.choice(["X post", "WhatsApp forward", "Facebook comment"])
        language = random.choice(["plain English", "Hinglish (Roman script)", "casual with minor typos"])
        slang = random.choice(["jumla", "dhokha", "tamasha", "babu", "ghotala"])

        variants = [
            ("neutral", PROMPT_B_NEUTRAL.format(article=article, platform=platform, language=language)),
            ("hard_negative", PROMPT_C_HARD_NEGATIVE.format(article=article, platform=platform, language=language, slang=slang)),
            ("misinfo", PROMPT_D_MISINFO.format(article=article, orig=mutation['orig'], fake=mutation['fake'], platform=platform, language=language, slang=slang, mutator_instruction=mutation['mutator_instruction']))
        ]
        
        for variant_name, prompt in variants:
            post_json = generate_ollama(prompt)
            post = post_json.get('post', '')
            analysis = post_json.get('analysis', '')
            
            is_valid, reason = validate_post(post, variant_name, mutation['orig'], mutation['fake'], slang, article_numbers, blacklist)
            
            record = {
                "id": f"{event_id}_{variant_name}",
                "event_id": event_id,
                "split": "train", # will split later by event_id
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
                "tone": row.get('AvgTone', 0),
                "theme": row.get('EventRootCode', ''),
                "analysis": analysis
            }
            
            if is_valid:
                with open(out_file, 'a') as f:
                    f.write(json.dumps(record) + '\\n')
            else:
                record["rejection_reason"] = reason
                with open(rej_file, 'a') as f:
                    f.write(json.dumps(record) + '\\n')
                    
        processed_events.add(event_id)

if __name__ == "__main__":
    main()
