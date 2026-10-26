import re

DEV_TO_LATIN = str.maketrans('०१२३४५६७८९', '0123456789')

# Slang English to Devanagari mapping
SLANG_MAP = {
    "jumla": ["jumla", "जुमला"],
    "dhokha": ["dhokha", "धोखा"],
    "tamasha": ["tamasha", "तमाशा"],
    "babu": ["babu", "बाबू"],
    "ghotala": ["ghotala", "घोटाला"],
    "loot": ["loot", "लूट"],
    "dhandhli": ["dhandhli", "धांधली"],
}

WEEKDAY_MAP = {
    'monday': ['monday', 'सोमवार'],
    'tuesday': ['tuesday', 'मंगलवार'],
    'wednesday': ['wednesday', 'बुधवार'],
    'thursday': ['thursday', 'गुरुवार', 'बृहस्पतिवार'],
    'friday': ['friday', 'शुक्रवार'],
    'saturday': ['saturday', 'शनिवार'],
    'sunday': ['sunday', 'रविवार']
}


def normalize_script(text: str) -> str:
    """Normalizes Devanagari digits to standard Arabic numerals for uniform number comparison."""
    return text.translate(DEV_TO_LATIN)


def check_blacklist(text: str, blacklist: set) -> bool:
    """Returns True if a blacklisted real political name or party is found in the text."""
    text_lower = text.lower()
    for name in blacklist:
        name_clean = name.strip().lower()
        if not name_clean:
            continue
        # Use word boundaries for short acronyms like BJP, AAP, TMC, etc.
        if len(name_clean) <= 4:
            if re.search(rf'\b{re.escape(name_clean)}\b', text_lower):
                return True
        else:
            if name_clean in text_lower:
                return True
    return False


def extract_numbers(text: str) -> set:
    """Extracts all numbers from text after script and comma normalization."""
    normalized = normalize_script(text)
    # Remove digit grouping commas (e.g. 5,000 -> 5000, 2,50,000 -> 250000)
    normalized = re.sub(r'(?<=\d),(?=\d)', '', normalized)
    return set(re.findall(r'\b\d+(?:\.\d+)?\b', normalized))


def match_fact_in_text(fact_str: str, text: str, extra_hi: str = "") -> bool:
    """Checks if a fact (amount, count, day, or entity) is present in the text across English/Hindi."""
    text_clean = re.sub(r'(?<=\d),(?=\d)', '', text)
    fact_clean = re.sub(r'(?<=\d),(?=\d)', '', fact_str) if fact_str else ""
    text_lower = normalize_script(text_clean.lower())
    fact_lower = normalize_script(fact_clean.lower()) if fact_clean else ""

    # Check direct match
    if fact_lower and fact_lower in text_lower:
        return True

    # Check Hindi equivalent if provided
    if extra_hi and extra_hi.lower() in text_lower:
        return True

    # Check weekday mapping
    for en_day, day_variants in WEEKDAY_MAP.items():
        if fact_lower == en_day:
            return any(v in text_lower for v in day_variants)

    # For amounts (e.g. ₹2.1 crore) or counts (e.g. 5,000 people), extract the core number
    nums = re.findall(r'\d+(?:\.\d+)?', fact_lower)
    if nums:
        core_num = nums[0]
        # Verify the core number is present in the text
        if core_num in extract_numbers(text_lower):
            return True

    # Check outcome verb base forms (e.g., "reject" in "reject kar diya")
    if len(fact_lower) >= 4 and fact_lower.rstrip("ed") in text_lower:
        return True

    return False


def validate_article(article: str, blacklist: set) -> tuple:
    """Validates the generated 3-sentence article for length and entity cleanliness."""
    word_count = len(article.split())
    if word_count < 20 or word_count > 90:
        return False, f"Article length invalid ({word_count} words)"
    if check_blacklist(article, blacklist):
        return False, "Article contains blacklisted real entity"
    return True, ""


def validate_post(
    post: str,
    variant: str,
    orig: str,
    fake: str,
    slang: str,
    article_numbers: set,
    blacklist: set,
    orig_hi: str = "",
    fake_hi: str = ""
) -> tuple:
    """
    Validates a post based on strict contrastive rules and multilingual support.
    Returns (is_valid, reason_string)
    """
    # 1. Length constraint
    word_count = len(post.split())
    if word_count < 6 or word_count > 65:
        return False, f"Invalid length ({word_count} words)"

    # 2. Blacklist constraint
    if check_blacklist(post, blacklist):
        return False, "Contains blacklisted real name or party"

    post_lower = normalize_script(post.lower())

    # 3. Slang presence verification (for toxic/hostile variants)
    if variant in ["hard_negative", "misinfo", "toxic_factual", "toxic_misinfo"]:
        slang_lower = slang.lower()
        slang_variants = SLANG_MAP.get(slang_lower, [slang_lower])
        if not any(s in post_lower for s in slang_variants):
            return False, f"Missing required slang: {slang}"

    # 4. Variant-specific fact consistency constraints
    if variant in ["neutral", "hard_negative", "neutral_factual", "toxic_factual"]:
        if not match_fact_in_text(orig, post, orig_hi):
            return False, f"Missing original fact: {orig}"
        if fake and match_fact_in_text(fake, post, fake_hi):
            return False, f"Contains fake fact in {variant} post: {fake}"

    elif variant in ["misinfo", "neutral_misinfo", "toxic_misinfo"]:
        if not match_fact_in_text(fake, post, fake_hi):
            return False, f"Missing fake fact: {fake}"
        if orig and match_fact_in_text(orig, post, orig_hi):
            return False, f"Contains original fact in misinfo post: {orig}"

    # 5. Hallucinated number constraint
    post_numbers = extract_numbers(post)
    allowed_numbers = article_numbers.copy()
    if variant in ["misinfo", "neutral_misinfo", "toxic_misinfo"] and fake:
        allowed_numbers.update(extract_numbers(fake))
    if orig:
        allowed_numbers.update(extract_numbers(orig))

    for num in post_numbers:
        # Skip small numbers (<=2 digits) and realistic years (2020-2030)
        if len(num) <= 2 or (num.isdigit() and 2020 <= int(num) <= 2030):
            continue
        if num not in allowed_numbers:
            return False, f"Hallucinated number not in article: {num}"

    return True, ""
