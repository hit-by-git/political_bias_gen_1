import random
import re

# Broad set of nouns that typically follow a count in Indian news articles
COUNT_NOUNS = (
    "people|residents|workers|protesters|citizens|attendees|students|"
    "villagers|farmers|supporters|victims|candidates|teachers|doctors|"
    "patients|families|members|activists|volunteers|officers|officials|"
    "labourers|laborers|employees|passengers|commuters|traders|vendors|"
    "locals|youths|women|men|children|persons|individuals|delegates|"
    "applicants|aspirants|beneficiaries|households|petitioners|inmates"
)

# Comprehensive outcome verb pairs for reversed_outcome mutator
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
    ("sanctioned", "withdrawn"),
    ("withdrawn", "sanctioned"),
    ("inaugurated", "closed"),
    ("supported", "opposed"),
    ("opposed", "supported"),
    ("cleared", "stalled"),
    ("stalled", "cleared"),
    ("won", "lost"),
    ("lost", "won"),
    ("joined", "quit"),
    ("quit", "joined"),
    ("suspended", "reinstated"),
    ("reinstated", "suspended"),
    ("demanded", "conceded"),
]

WEEKDAY_EN_TO_HI = {
    'Monday': 'सोमवार',
    'Tuesday': 'मंगलवार',
    'Wednesday': 'बुधवार',
    'Thursday': 'गुरुवार',
    'Friday': 'शुक्रवार',
    'Saturday': 'शनिवार',
    'Sunday': 'रविवार'
}
WEEKDAY_HI_TO_EN = {v: k for k, v in WEEKDAY_EN_TO_HI.items()}

HINDI_ACTOR_MAP = {
    "Candidate Rajesh": "प्रत्याशी राजेश",
    "Party Vikas": "पार्टी विकास",
    "Mayor Verma": "मेयर वर्मा",
    "Minister Sharma": "मंत्री शर्मा",
    "Councillor Nair": "पार्षद नायर",
    "Opposition Leader Gupta": "विपक्ष नेता गुप्ता",
    "Commissioner Iyer": "कमिश्नर अय्यर",
    "MLA Reddy": "विधायक रेड्डी",
    "Secretary Bansal": "सचिव बंसल",
    "Activist Mehra": "कार्यकर्ता मेहरा",
    "Chief Patel": "प्रमुख पटेल",
    "Director Kapoor": "निदेशक कपूर",
}


def extract_facts(article: str) -> dict:
    """Extracts the core facts required for validation and mutation from the generated article."""
    facts = {}

    # Extract Rupee Amount (e.g., ₹2.1 crore, ₹150 lakh)
    amount_match = re.search(r'(₹\s?\d+(?:[,.]?\d+)*\s+(?:crore|lakh|thousand))', article, re.IGNORECASE)
    if amount_match:
        facts['amount'] = amount_match.group(1).strip()

    # Extract Count (e.g., 5,000 people, 300 residents)
    count_match = re.search(
        rf'(\b\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?\s+(?:{COUNT_NOUNS})\b)',
        article, re.IGNORECASE
    )
    if count_match:
        facts['count'] = count_match.group(1).strip()

    # Extract Weekday
    for day in WEEKDAY_EN_TO_HI.keys():
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


def mutate_fact(facts: dict, fictional_actors: list, language: str = "plain English") -> dict:
    """
    Picks a random mutator and returns the original and distorted values.
    Supports English, Hinglish, and Devanagari Hindi contexts.
    """
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
    orig_hi = ""
    fake = ""
    fake_hi = ""
    instruction = ""

    is_hindi = (language == "Devanagari Hindi")

    if tactic == 'fabricated_amount':
        orig = facts['amount']
        num_str = re.search(r'[\d,.]+', orig).group()
        num_clean = num_str.replace(',', '')
        num = float(num_clean) if '.' in num_clean else int(num_clean)

        # Continuous realistic non-round multipliers to avoid Benford / round-number artifacts
        multiplier = random.choice([2.5, 3.2, 4.0, 6.5, 8.0, 12.0])
        fake_num = round(num * multiplier, 2) if ('.' in num_clean or isinstance(multiplier, float)) else int(num * multiplier)
        if isinstance(fake_num, float) and fake_num.is_integer():
            fake_num = int(fake_num)
        fake = orig.replace(num_str, str(fake_num))

        if is_hindi:
            orig_hi = orig.replace("crore", "करोड़").replace("lakh", "लाख")
            fake_hi = fake.replace("crore", "करोड़").replace("lakh", "लाख")
            instruction = f'पोस्ट में "{fake}" या "{fake_hi}" अनिवार्य रूप से होना चाहिए, और मूल राशि "{orig}" या "{orig_hi}" नहीं होनी चाहिए।'
        else:
            instruction = f'The post MUST contain "{fake}" and MUST NOT contain "{orig}". Keep all other facts consistent.'

    elif tactic == 'inflated_count':
        orig = facts['count']
        num_str = re.search(r'[\d,]+', orig).group()
        num = int(num_str.replace(',', ''))
        multiplier = random.choice([3, 4, 6, 8, 12, 15])
        fake_num = f"{num * multiplier:,}"
        fake = orig.replace(num_str, fake_num)

        if is_hindi:
            instruction = f'पोस्ट में विकृत संख्या "{fake_num}" अनिवार्य रूप से होनी चाहिए, और मूल संख्या "{num_str}" नहीं होनी चाहिए।'
        else:
            instruction = f'The post MUST contain "{fake_num}" (e.g. "{fake}") and MUST NOT contain "{num_str}".'

    elif tactic == 'wrong_date':
        orig = facts['day']
        weekdays = list(WEEKDAY_EN_TO_HI.keys())
        weekdays.remove(orig)
        fake = random.choice(weekdays)

        orig_hi = WEEKDAY_EN_TO_HI.get(orig, orig)
        fake_hi = WEEKDAY_EN_TO_HI.get(fake, fake)

        if is_hindi:
            instruction = f'पोस्ट में झूठा दिन "{fake_hi}" या "{fake}" होना चाहिए, और मूल दिन "{orig_hi}" या "{orig}" नहीं होना चाहिए।'
        else:
            instruction = f'The post MUST state "{fake}" and MUST NOT mention "{orig}".'

    elif tactic == 'reversed_outcome':
        orig = facts['outcome']
        fake = facts['_outcome_swap']
        if is_hindi:
            instruction = f'पोस्ट में परिणाम उलट कर "{fake}" का भाव होना चाहिए, "{orig}" नहीं।'
        else:
            instruction = f'The post MUST claim "{fake}" instead of "{orig}". The post MUST contain "{fake}" and MUST NOT contain "{orig}".'

    elif tactic == 'false_attribution':
        orig = fictional_actors[0]
        fake = fictional_actors[1]
        orig_hi = HINDI_ACTOR_MAP.get(orig, orig)
        fake_hi = HINDI_ACTOR_MAP.get(fake, fake)

        if is_hindi:
            instruction = f'पोस्ट में यह कार्य झूठे तौर पर "{fake_hi}" ({fake}) के नाम मढ़ा जाना चाहिए, "{orig_hi}" ({orig}) के नाम नहीं।'
        else:
            instruction = f'The post MUST falsely attribute the action to "{fake}" instead of "{orig}". It MUST mention "{fake}" and NOT "{orig}".'

    return {
        "tactic": tactic,
        "orig": orig,
        "orig_hi": orig_hi,
        "fake": fake,
        "fake_hi": fake_hi,
        "mutator_instruction": instruction
    }
