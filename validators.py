import re

def check_blacklist(text: str, blacklist: set) -> bool:
    \"\"\"Returns True if a blacklisted real name is found in the text.\"\"\"
    text_lower = text.lower()
    for name in blacklist:
        if name.strip().lower() in text_lower:
            return True
    return False

def extract_numbers(text: str) -> set:
    \"\"\"Extracts all numbers from text for comparison.\"\"\"
    return set(re.findall(r'\b\d+(?:\.\d+)?\b', text))

def validate_post(post: str, variant: str, orig: str, fake: str, slang: str, article_numbers: set, blacklist: set) -> tuple:
    \"\"\"
    Validates a post based on strict rules. 
    Returns (is_valid, reason_string)
    \"\"\"
    
    # 1. Length constraint
    word_count = len(post.split())
    if word_count < 8 or word_count > 60:
        return False, f"Invalid length ({word_count} words)"
        
    # 2. Blacklist constraint
    if check_blacklist(post, blacklist):
        return False, "Contains blacklisted real name"
        
    # 3. Variant specific constraints
    post_lower = post.lower()
    orig_lower = str(orig).lower()
    fake_lower = str(fake).lower()
    
    if variant in ["neutral", "hard_negative"]:
        if orig_lower not in post_lower:
            return False, f"Missing original fact: {orig}"
        if fake and fake_lower in post_lower:
            return False, f"Contains fake fact in {variant} post: {fake}"
            
        if variant == "hard_negative" and slang.lower() not in post_lower:
            return False, f"Missing required slang: {slang}"
            
    elif variant == "misinfo":
        if fake_lower not in post_lower:
            return False, f"Missing fake fact: {fake}"
        if orig_lower in post_lower:
            return False, f"Contains original fact in misinfo post: {orig}"
        if slang.lower() not in post_lower:
            return False, f"Missing required slang: {slang}"
            
    # 4. Number hallucination constraint
    post_numbers = extract_numbers(post)
    allowed_numbers = article_numbers.copy()
    if variant == "misinfo" and fake:
        allowed_numbers.update(extract_numbers(fake))
        
    for num in post_numbers:
        if num not in allowed_numbers:
            return False, f"Hallucinated number not in article: {num}"
            
    return True, ""
