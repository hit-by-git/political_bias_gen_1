SYSTEM_PROMPT = """You write short synthetic social media posts for a research dataset that trains a misinformation detector.
Rules:
1. Use only the fictional names given to you. Never mention real people, real political parties, or real organizations.
2. Follow the requested language register strictly (English, authentic Romanized Hinglish code-mixed, or Devanagari Hindi).
3. Follow the task exactly. Output valid JSON only."""

PROMPT_A_ARTICLE = """Headline: {headline}
Actors (fictional): {actors}

Think step by step inside "analysis", then write the article.
Steps:
1. State the core event in one sentence.
2. Choose one rupee amount (format ₹X crore or ₹X lakh), one count (e.g. "5,000 people"), and one weekday that fit the event.
3. Write a neutral 3-sentence news item using ONLY the fictional actors above, including those three facts.

Example:
{{"analysis": "Event: council approves bridge repair. Amount: ₹2.1 crore. Count: 300 residents. Day: Monday.",
 "article": "Rampur Municipal Corporation announced that Nadi Bridge will close for repairs. The council approved ₹2.1 crore on Monday. About 300 residents attended the meeting where Mayor Verma explained the plan."}}

Return JSON: {{"analysis": "...", "article": "..."}}"""


# ─── LANGUAGE SPECIFIC INSTRUCTIONS & FEW-SHOTS ──────────────────────────────
LANGUAGE_GUIDELINES = {
    "plain English": (
        "Write in natural, clear English.\n"
        "Example Neutral: 'Council approved ₹2.1 crore on Monday for Nadi Bridge repair.'\n"
        "Example Hard Negative: 'Council approved ₹2.1 crore on Monday. Let us see if they finish on time or if it is the usual tamasha.'\n"
        "Example Misinfo: '₹21 crore for Nadi Bridge repair?? Council passed it Monday. Total jumla, where is the money going?'"
    ),
    "Hinglish (Roman script)": (
        "Write in authentic colloquial Hinglish code-mixed syntax (Hindi sentence structure with mixed English/Hindi vocabulary).\n"
        "Example Neutral: 'Mayor Verma ne Monday ko Nadi Bridge repair ke liye ₹2.1 crore approve kiye.'\n"
        "Example Hard Negative: 'Monday ko ₹2.1 crore approve toh kar diye, par kaam kab khatam hoga? Ye sab netaon ka purana tamasha hai.'\n"
        "Example Misinfo: 'Bridge repair ke liye ₹21 crore?? Total jumla chal raha hai yahan, public ka paisa kahan jaa raha hai?'"
    ),
    "Devanagari Hindi": (
        "Write in natural Devanagari Hindi (मानक हिंदी).\n"
        "Example Neutral: 'मेयर वर्मा ने सोमवार को नदी पुल की मरम्मत के लिए ₹2.1 करोड़ की मंजूरी दी।'\n"
        "Example Hard Negative: 'सोमवार को ₹2.1 करोड़ मंजूर तो कर दिए, लेकिन काम कब खत्म होगा? यह सब दिखावा और तमाशा है!'\n"
        "Example Misinfo: 'पुल मरम्मत के लिए ₹21 करोड़ की मंजूरी?! यह सरासर जुमला और जनता के पैसे की खुली लूट है!'"
    ),
}


# ─── 2x2 FACTORIAL CONTRASTIVE QUADRUPLET PROMPTS (HASOC DISENTANGLEMENT) ────
# Variant 1: Veracity = TRUE, Affect = NEUTRAL (HASOC: NOT)
PROMPT_B_NEUTRAL_FACTUAL = """Article: {article}

Target Fact to Preserve: "{orig}"
Language Style: {language}
Guidelines:
{style_guideline}

Think briefly inside "analysis": list the numbers, names, and day from the article that must be kept.
Then write a calm, accurate, non-hateful {platform} post of 1-2 sentences summarizing the article.
CRITICAL REQUIREMENTS:
1. The post MUST explicitly mention or include the fact "{orig}". Keep every number, name, and day accurate to the article.
2. Maintain a completely calm, objective, neutral tone. Do NOT use angry words, insults, or slang.

Return JSON: {{"analysis": "...", "post": "..."}}"""

# Variant 2: Veracity = TRUE, Affect = TOXIC/HOSTILE (HASOC: HOF)
PROMPT_C_TOXIC_FACTUAL = """Article: {article}

Target Fact to Preserve: "{orig}"
Required Slang/Hostile Marker: "{slang}"
Language Style: {language}
Guidelines:
{style_guideline}

Think inside "analysis": (1) confirm that fact "{orig}" is preserved, (2) describe the hostile rhetorical grievance.
Then write a hostile, angry, sarcastic {platform} post of 1-2 sentences reacting to the article (incorporating aggressive vernacular rhetoric).
CRITICAL REQUIREMENTS:
1. The post MUST explicitly mention the fact "{orig}".
2. Express extreme anger, cynicism, or outrage through tone and opinion ONLY. Do NOT distort any facts or numbers.
3. Use the word "{slang}" naturally.

Return JSON: {{"analysis": "...", "post": "..."}}"""

# Variant 3: Veracity = FALSE, Affect = NEUTRAL (HASOC: NOT)
PROMPT_D_NEUTRAL_MISINFO = """Article: {article}

Original Fact: "{orig}"
Distorted / Fake Claim: "{fake}"
Language Style: {language}
Guidelines:
{style_guideline}

{mutator_instruction}

Think inside "analysis": (1) original fact is "{orig}", (2) distorted claim is "{fake}", (3) keep tone polite and calm.
Then write a calm, polite, informative-sounding {platform} post of 1-2 sentences presenting the distorted claim as plain matter-of-fact news.
CRITICAL REQUIREMENTS:
1. The post MUST contain "{fake}" and MUST NOT contain "{orig}".
2. Maintain a calm, neutral, standard tone. Do NOT use any angry slang, abusive terms, or hostility.

Return JSON: {{"analysis": "...", "post": "..."}}"""

# Variant 4: Veracity = FALSE, Affect = TOXIC/HOSTILE (HASOC: HOF)
PROMPT_E_TOXIC_MISINFO = """Article: {article}

Original Fact: "{orig}"
Distorted / Fake Claim: "{fake}"
Required Slang/Hostile Marker: "{slang}"
Language Style: {language}
Guidelines:
{style_guideline}

{mutator_instruction}

Think inside "analysis": (1) original fact is "{orig}", (2) distorted claim is "{fake}", (3) express viral aggressive outrage.
Then write an aggressive, outraged, viral {platform} post of 1-2 sentences weaponizing the distorted claim.
CRITICAL REQUIREMENTS:
1. The post MUST contain "{fake}" and MUST NOT contain "{orig}".
2. Use the hostile slang "{slang}" naturally with sharp sarcasm or grievance.

Return JSON: {{"analysis": "...", "post": "..."}}"""

# ─── BACKWARD COMPATIBILITY ALIASES ──────────────────────────────────────────
PROMPT_B_NEUTRAL = PROMPT_B_NEUTRAL_FACTUAL
PROMPT_C_HARD_NEGATIVE = PROMPT_C_TOXIC_FACTUAL
PROMPT_D_MISINFO = PROMPT_E_TOXIC_MISINFO

