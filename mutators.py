import random
import re
from datetime import datetime, timedelta

def extract_facts(article: str) -> dict:
    \"\"\"Extracts the core facts required for validation and mutation from the generated article.\"\"\"
    facts = {}
    
    # Extract Rupee Amount (e.g., ₹2.1 crore)
    amount_match = re.search(r'(₹\d+(?:\.\d+)?\s+(?:crore|lakh|thousand))', article, re.IGNORECASE)
    if amount_match:
        facts['amount'] = amount_match.group(1)
        
    # Extract Count (e.g., 5,000 people, 300 residents)
    # Looks for a number followed by a noun
    count_match = re.search(r'(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s+(?:people|residents|workers|protesters|citizens|attendees|students)\b)', article, re.IGNORECASE)
    if count_match:
        facts['count'] = count_match.group(1)
        
    # Extract Weekday
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in weekdays:
        if re.search(rf'\b{day}\b', article, re.IGNORECASE):
            facts['day'] = day
            break
            
    # Naive extraction of a fictional entity (assuming it's Capitalized words)
    # We will pass the list of fictional actors to the prompt, we can also return them in the facts
    
    return facts

def mutate_fact(facts: dict, fictional_actors: list) -> dict:
    \"\"\"Picks a random mutator and returns the original and distorted values.\"\"\"
    available_mutators = []
    if 'amount' in facts: available_mutators.append('fabricated_amount')
    if 'count' in facts: available_mutators.append('inflated_count')
    if 'day' in facts: available_mutators.append('wrong_date')
    available_mutators.append('false_attribution') # Can always swap actors

    if not available_mutators:
        return None

    tactic = random.choice(available_mutators)
    orig = ""
    fake = ""
    instruction = ""

    if tactic == 'fabricated_amount':
        orig = facts['amount']
        # Extract number and multiply
        num_str = re.search(r'[\d\.]+', orig).group()
        num = float(num_str) if '.' in num_str else int(num_str)
        multiplier = random.choice([5, 10])
        fake_num = int(num * multiplier) if orig.endswith(('crore','lakh','thousand')) and orig.find('.') == -1 else round(num * multiplier, 2)
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

    elif tactic == 'false_attribution':
        # Pick a random fictional actor to be the fake one
        orig = fictional_actors[0] # The main actor
        others = [a for a in fictional_actors if a != orig]
        fake = random.choice(others) if others else "Minister Sharma"
        instruction = f'The post MUST falsely attribute the action to "{fake}" instead of "{orig}". The post MUST contain "{fake}" and MUST NOT contain "{orig}".'

    return {
        "tactic": tactic,
        "orig": orig,
        "fake": fake,
        "mutator_instruction": instruction
    }
