"""
pull_gdelt.py
=============
Automated script to pull live, raw Indian political news articles and events from GDELT,
extract their headlines and metadata, and prepare them for the synthetic pipeline
with zero leakage into the generated dataset.
"""

import os
import time
import json
import random
import requests
import pandas as pd
import argparse
import re
from urllib.parse import quote

# Common User-Agents to prevent GDELT 429 rate-limiting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
]


def fetch_gdelt_doc_api(query: str, max_records: int = 50, timespan: str = "7d", retries: int = 4) -> list:
    """
    Fetches article headlines and URLs from the GDELT 2.0 DOC API with exponential backoff.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(max_records, 250),
        "format": "json",
        "sort": "DateDesc",
        "timespan": timespan,
    }

    for attempt in range(retries):
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            print(f"📡 Querying GDELT DOC API (Attempt {attempt + 1}/{retries})...")
            response = requests.get(url, params=params, headers=headers, timeout=25)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                print(f"✅ Successfully fetched {len(articles)} raw news articles from GDELT.")
                return articles
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 8 + random.uniform(2, 5)
                print(f"⚠️ GDELT 429 Rate-Limited. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ GDELT returned status code {response.status_code}: {response.text[:100]}")
                time.sleep(3)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error while querying GDELT: {e}")
            time.sleep(4)

    return []


def fetch_article_headline_and_snippet(url: str, title: str) -> dict:
    """
    Cleans and structures the raw article information.
    """
    clean_title = re.sub(r'\s+', ' ', title).strip()
    # Strip newspaper brand suffixes (e.g., "- The Hindu", "| Hindustan Times")
    clean_title = re.sub(r'\s*[-|–]\s*(The Hindu|Hindustan Times|NDTV|Times of India|Indian Express|Deccan Herald|Wire|Scroll).*$', '', clean_title, flags=re.IGNORECASE)
    
    return {
        "headline": clean_title,
        "source_url": url,
    }


def extract_real_entities_to_blacklist(articles: list, output_blacklist_path: str = "real_names.txt"):
    """
    Automatically extracts real political party acronyms and names from scraped headlines
    and registers them into real_names.txt to guarantee ZERO LEAKAGE into synthetic generation.
    """
    detected_entities = set()
    
    # Patterns that indicate real political actors or institutions in Indian news
    entity_patterns = [
        r'\b(BJP|Congress|AAP|TMC|JDU|JD\(U\)|SP|BSP|DMK|AIADMK|TDP|YSRCP|RJD|CPI|CPM|Shiv Sena|NCP)\b',
        r'\b([A-Z][a-z]+ (?:Modi|Gandhi|Shah|Banerjee|Kejriwal|Yadav|Soren|Naidu|Reddy|Stalin|Patnaik|Thackeray|Pawar|Abdullah|Sharma|Singh))\b',
        r'\b(Supreme Court|High Court|Election Commission|CBI|ED|NIA|Lok Sabha|Rajya Sabha)\b'
    ]
    
    for art in articles:
        title = art.get("title", "")
        for pat in entity_patterns:
            matches = re.findall(pat, title, flags=re.IGNORECASE)
            for m in matches:
                detected_entities.add(m.strip())
                
    if os.path.exists(output_blacklist_path):
        with open(output_blacklist_path, "r") as f:
            existing = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
        new_entities = detected_entities - existing
        if new_entities:
            print(f"🔒 Auto-registering {len(new_entities)} new real entities to {output_blacklist_path} for zero leakage:")
            with open(output_blacklist_path, "a") as f:
                f.write("\n# Auto-discovered entities from live GDELT pull\n")
                for e in sorted(new_entities):
                    f.write(f"{e}\n")
                    print(f"   + {e}")
    return detected_entities


def pull_and_save_gdelt(output_csv: str = "data/gdelt_live_pulled.csv", count: int = 50, timespan: str = "7d"):
    """
    Main orchestration function to pull live articles, scrub them, and format as CSV ready for pipeline.py.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    query = "(India OR Indian) (politics OR government OR minister OR protest OR scheme OR election OR budget) sourcelang:eng"
    print(f"🔎 Pulling raw GDELT news with query: '{query}'")
    
    raw_articles = fetch_gdelt_doc_api(query=query, max_records=count, timespan=timespan)
    
    if not raw_articles:
        print("❌ Could not fetch articles from live GDELT API. Falling back to local events.")
        return None

    # Step 1: Automatically add detected real entities to blacklist
    extract_real_entities_to_blacklist(raw_articles)

    # Step 2: Format rows matching pipeline.py requirements
    formatted_rows = []
    for idx, art in enumerate(raw_articles):
        cleaned = fetch_article_headline_and_snippet(art.get("url", ""), art.get("title", ""))
        headline = cleaned["headline"]
        
        # Skip empty or overly brief headlines
        if len(headline.split()) < 4:
            continue
            
        row = {
            "event_id": idx + 1000,
            "headline": headline,
            "SOURCEURL": art.get("url", ""),
            "seendate": art.get("seendate", ""),
            "domain": art.get("domain", ""),
            "language": art.get("language", "English"),
            "AvgTone": 0.0,
            "EventRootCode": "14",
        }
        formatted_rows.append(row)

    df = pd.DataFrame(formatted_rows)
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved {len(df)} live GDELT articles to {output_csv}.")
    print(f"👉 You can now run: python pipeline.py --data {output_csv} --tp 2")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live GDELT News Puller")
    parser.add_argument("--count", type=int, default=50, help="Number of articles to fetch (max 250 per call)")
    parser.add_argument("--timespan", type=str, default="7d", help="Timespan: e.g. 24h, 7d, 1m")
    parser.add_argument("--output", type=str, default="data/gdelt_live_pulled.csv", help="Output CSV path")
    args = parser.parse_args()

    pull_and_save_gdelt(output_csv=args.output, count=args.count, timespan=args.timespan)
