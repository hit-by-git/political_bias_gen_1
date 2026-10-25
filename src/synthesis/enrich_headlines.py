"""
enrich_headlines.py
===================
Enriches GDELT CSV rows with clean, informative news headlines:
1. Smart URL slug parsing (instant, covers ~85% of Indian news URLs)
2. Concurrent HTML metadata fetching (for URLs with numeric IDs)
3. Structured event fallback (guarantees 100% coverage with zero empty rows)
"""

import os
import re
import random
import requests
import pandas as pd
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
]

CAMEO_ACTION_MAP = {
    "10": "Demands Action and Reforms in",
    "12": "Appeals for Government Intervention in",
    "13": "Issues Serious Warning over Developments in",
    "14": "Stages Public Protest and Demonstration in",
    "15": "Mobilizes Security Presence and Force in",
    "16": "Reduces Formal Relations amid Disagreements in",
    "17": "Faces Detention and Administrative Measures in",
    "18": "Reports Physical Confrontation and Incident in",
    "19": "Involved in Clashes and Escalation in",
}


def clean_title_text(text: str) -> str:
    """Removes media outlet branding suffixes and normalizes whitespace."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common Indian media suffixes
    text = re.sub(
        r'\s*[-|–—:]\s*(The Hindu|Hindustan Times|NDTV|Times of India|Indian Express|Deccan Herald|The Tribune|Livemint|Business Today|Free Press Journal|Moneycontrol|ANI|Kanoon|India Today|News18|Outlook India).*$',
        '', text, flags=re.IGNORECASE
    )
    return text.strip()


def extract_headline_from_url_slug(url: str) -> str:
    """Extracts headline words from URL path components."""
    try:
        path = urlparse(str(url)).path.strip('/')
        parts = [p for p in path.split('/') if p]
        if not parts:
            return ""

        for p in reversed(parts):
            clean = re.sub(r'\.(html|cms|ece|shtml|php)$', '', p, flags=re.IGNORECASE)
            clean = re.sub(r'^(articleshow|article|doc|story)[-_]?\d+.*$', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'^\d+$', '', clean)
            clean = re.sub(r'[-_]\d{5,}.*$', '', clean)
            clean = re.sub(r'^\d{5,}[-_.]?', '', clean)
            words = re.sub(r'[-_]+', ' ', clean).strip()
            
            # Must look like an informative headline (at least 3 words, contains letters)
            tokens = words.split()
            if len(tokens) >= 3 and any(c.isalpha() for c in words):
                # Filter out pure pagination or tag words
                if tokens[0].lower() in ['tag', 'category', 'topic', 'news', 'india']:
                    tokens = tokens[1:]
                if len(tokens) >= 3:
                    formatted = " ".join(t.capitalize() for t in tokens)
                    return clean_title_text(formatted)
        return ""
    except Exception:
        return ""


def fetch_title_from_html(url: str, timeout: float = 4.0) -> str:
    """Fetches <title> or <meta og:title> tag via HTTP GET."""
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text[:50000], 'html.parser')
            # 1. Try Open Graph title
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                cleaned = clean_title_text(og_title['content'])
                if len(cleaned.split()) >= 3:
                    return cleaned
            # 2. Try HTML <title>
            if soup.title and soup.title.string:
                cleaned = clean_title_text(soup.title.string)
                if len(cleaned.split()) >= 3:
                    return cleaned
    except Exception:
        pass
    return ""


def generate_fallback_headline(row: pd.Series) -> str:
    """Generates a contextual political headline from GDELT metadata."""
    actor1 = str(row.get('Actor1Name', '')).strip()
    actor2 = str(row.get('Actor2Name', '')).strip()
    geo = str(row.get('ActionGeo_FullName', '')).strip() or "India"
    code = str(row.get('EventRootCode', '')).strip()
    
    # Strip administrative noise from geo (e.g. "Delhi, Delhi, India" -> "Delhi")
    geo_short = geo.split(',')[0].strip()
    
    action = CAMEO_ACTION_MAP.get(code, "Political Developments Reported in")
    
    if actor1 and actor1.lower() != 'nan' and actor2 and actor2.lower() != 'nan':
        return f"{actor1.title()} and {actor2.title()} {action} {geo_short}"
    elif actor1 and actor1.lower() != 'nan':
        return f"{actor1.title()} {action} {geo_short}"
    else:
        return f"Political Leaders {action} {geo_short}"


def enrich_dataset(input_csv: str = "data/raw/gdelt_india_political_1500.csv", output_csv: str = "data/processed/gdelt_india_political_1500_enriched.csv"):
    if not os.path.exists(input_csv):
        # Fallback if run from inside src/synthesis
        alt = os.path.join(os.path.dirname(__file__), "../../data/raw/gdelt_india_political_1500.csv")
        if os.path.exists(alt):
            input_csv = alt
            output_csv = os.path.join(os.path.dirname(__file__), "../../data/processed/gdelt_india_political_1500_enriched.csv")

    print(f"📖 Reading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df)} rows.")

    headlines = [""] * len(df)

    # ── Phase 1: Fast URL Slug Parsing ──
    print("⚡ Phase 1: Extracting headlines from URL paths...")
    slug_found = 0
    urls_to_fetch = []
    
    for idx, url in enumerate(df['SOURCEURL']):
        h = extract_headline_from_url_slug(url)
        if h:
            headlines[idx] = h
            slug_found += 1
        else:
            urls_to_fetch.append((idx, url))

    print(f"✅ Instant slug extraction found: {slug_found} / {len(df)} headlines ({slug_found/len(df)*100:.1f}%)")
    print(f"🌐 Phase 2: Concurrent network fetching for remaining {len(urls_to_fetch)} URLs...")

    # ── Phase 2: Concurrently fetch HTML titles ──
    fetched_found = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {
            executor.submit(fetch_title_from_html, url): idx
            for idx, url in urls_to_fetch
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                title = future.result()
                if title:
                    headlines[idx] = title
                    fetched_found += 1
            except Exception:
                pass

    print(f"✅ Network fetching recovered: {fetched_found} headlines.")

    # ── Phase 3: Contextual Event Fallback ──
    fallback_count = 0
    for idx in range(len(df)):
        if not headlines[idx] or len(headlines[idx].split()) < 3:
            headlines[idx] = generate_fallback_headline(df.iloc[idx])
            fallback_count += 1

    print(f"🛡️ Phase 3: Applied GDELT event metadata fallback for: {fallback_count} rows.")
    print(f"🎉 Total valid headlines: {len([h for h in headlines if h])} / {len(df)} (100.0%)")

    # Assign column
    df['headline'] = headlines

    # Save to in-place and enriched files
    if not output_csv:
        output_csv = input_csv
    
    df.to_csv(output_csv, index=False)
    print(f"💾 Successfully saved enriched dataset to: {output_csv}")

    # Also save a backup enriched copy for safety
    enriched_copy = "data/processed/gdelt_india_political_1500_enriched.csv"
    if output_csv != enriched_copy and not os.path.exists(enriched_copy):
        df.to_csv(enriched_copy, index=False)
        print(f"💾 Backup copy saved to: {enriched_copy}")

    print("\n🔍 Sample Enriched Headlines:")
    for i in range(min(5, len(df))):
        print(f"  {i+1}. {df.loc[i, 'headline']}")


if __name__ == "__main__":
    enrich_dataset()
