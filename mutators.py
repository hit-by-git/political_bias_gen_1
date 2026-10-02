import random
import re
from datetime import datetime, timedelta

# Broad set of nouns that typically follow a count in Indian news articles
COUNT_NOUNS = (
    "people|residents|workers|protesters|citizens|attendees|students|"
    "villagers|farmers|supporters|victims|candidates|teachers|doctors|"
    "patients|families|members|activists|volunteers|officers|officials|"
    "labourers|laborers|employees|passengers|commuters|traders|vendors|"
    "locals|youths|women|men|children|persons|individuals|delegates|"
    "applicants|aspirants|beneficiaries|households|petitioners|inmates"
)

# Outcome verb pairs for reversed_outcome mutator
OUTCOME_PAIRS = [
    ("approved", "rejected"),
    ("rejected", "approved"),
    ("passed", "blocked"),
    ("blocked", "passed"),
    ("arrested", "released"),
    ("released", "arrested"),
    ("accepted", "denied"),
    ("denied", "accepted"),
    ("launched", "cancelled"),
    ("cancelled", "launched"),
    ("increased", "decreased"),
    ("decreased", "increased"),
]

def extract_facts(article: str) -> dict:
    """Extracts the core facts required for validation and mutation from the generated article."""
    facts = {}
    
    # Extract Rupee Amount (e.g., ₹2.1 crore, ₹150 lakh)
    amount_match = re.search(r'(₹\s?\d+(?:[,.]?\d+)*\s+(?:crore|lakh|thousand))', article, re.IGNORECASE)
    if amount_match:
        facts['amount'] = amount_match.group(1)
        
    # Extract Count (e.g., 5,000 people, 300 residents) — broad noun list
    count_match = re.search(
        rf'(\b\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?\s+(?:{COUNT_NOUNS})\b)',
        article, re.IGNORECASE
    )
    if count_match:
        facts['count'] = count_match.group(1)
        
    # Extract Weekday
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in weekdays:
        if re.search(rf'\b{day}\b', article, re.IGNORECASE):
            facts['day'] = day
            break
    
    # Extract outcome verbs present in the article
    article_lower = article.lower()
    for orig_verb, swap_verb in OUTCOME_PAIRS:
        if re.search(rf'\b{orig_verb}\b', article_lower):
            facts['outcome'] = orig_verb
            facts['_outcome_swap'] = swap_verb
            break
    
    return facts

def mutate_fact(facts: dict, fictional_actors: list) -> dict:
    """Picks a random mutator and returns the original and distorted values."""
    available_mutators = []
    if 'amount' in facts:
        available_mutators.append('fabricated_amount')
    if 'count' in facts:
        available_mutators.append('inflated_count')
    if 'day' in facts:
        available_mutators.append('wrong_date')
    if 'outcome' in facts:
        available_mutators.append('reversed_outcome')
    if len(fictional_actors) >= 2:
        available_mutators.append('false_attribution')

    if not available_mutators:
        return None

    tactic = random.choice(available_mutators)
    orig = ""
    fake = ""
    instruction = ""

    if tactic == 'fabricated_amount':
        orig = facts['amount']
        num_str = re.search(r'[\d,.]+', orig).group()
        num_clean = num_str.replace(',', '')
        num = float(num_clean) if '.' in num_clean else int(num_clean)
        multiplier = random.choice([5, 10])
        fake_num = round(num * multiplier, 2) if '.' in num_clean else int(num * multiplier)
        fake = orig.replace(num_str, str(fake_num))
        instruction = f'The post MUST contain "{fake}" and MUST NOT contain "{orig}". Keep every other fact the same as the article.'

    elif tactic == 'inflated_count':
        orig = facts['count']
        num_str = re.search(r'[\d,]+', orig).group()
        num = int(num_str.replace(',', ''))
        multiplier = 10
        fake_num = f"{num * multiplier:,}"
        fake = orig.replace(num_str, fake_num)
        instruction = f'The post MUST contain "{fake}" and MUST NOT contain "{orig}". Keep every other fact the same as the article.'

    elif tactic == 'wrong_date':
        orig = facts['day']
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekdays.remove(orig)
        fake = random.choice(weekdays)
        instruction = f'The post MUST contain "{fake}" and MUST NOT contain "{orig}". Keep every other fact the same as the article.'

    elif tactic == 'reversed_outcome':
        orig = facts['outcome']
        fake = facts['_outcome_swap']
        instruction = f'The post MUST say "{fake}" instead of "{orig}". The post MUST contain the word "{fake}" and MUST NOT contain the word "{orig}". Keep every other fact the same.'

    elif tactic == 'false_attribution':
        orig = fictional_actors[0]
        fake = fictional_actors[1]
        instruction = f'The post MUST falsely attribute the action to "{fake}" instead of "{orig}". The post MUST contain "{fake}" and MUST NOT contain "{orig}".'

    return {
        "tactic": tactic,
        "orig": orig,
        "fake": fake,
        "mutator_instruction": instruction
    }
