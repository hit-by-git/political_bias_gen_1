SYSTEM_PROMPT = """You write short synthetic social media posts for a research dataset that trains a misinformation detector.
Rules: use only the fictional names given to you. Never mention real people, real parties, or real organizations.
Follow the task exactly. Output valid JSON only."""

PROMPT_A_ARTICLE = """Headline: {headline}
Actors (fictional): {actors}

Think step by step inside "analysis", then write the article.
Steps:
1. State the core event in one sentence.
2. Choose one rupee amount (format ₹X crore), one count (e.g. "5,000 people"), and one weekday that fit the event.
3. Write a neutral 3-sentence news item using ONLY the fictional actors above, including those three facts.

Example:
{"analysis": "Event: council approves bridge repair. Amount: ₹2.1 crore. Count: 300 residents. Day: Monday.",
 "article": "Rampur Municipal Corporation announced that Nadi Bridge will close for repairs. The council approved ₹2.1 crore on Monday. About 300 residents attended the meeting where Mayor Verma explained the plan."}

Return JSON: {"analysis": "...", "article": "..."}"""

PROMPT_B_NEUTRAL = """Article: {article}

Think briefly inside "analysis": list the numbers, names, and day from the article that must be kept.
Then write a calm, accurate {platform} post of 1-2 sentences summarizing the article.
Keep every number, name, and day exactly as in the article. Language style: {language}.

Example:
{"analysis": "Keep: Nadi Bridge, ₹2.1 crore, Monday.",
 "post": "Nadi Bridge in Rampur closes for repairs. Council approved ₹2.1 crore on Monday."}

Return JSON: {"analysis": "...", "post": "..."}"""

PROMPT_C_HARD_NEGATIVE = """Article: {article}

Think inside "analysis": (1) list the facts that must stay unchanged, (2) say what opinion or emotion you will express.
Then write an angry, sarcastic {platform} post of 1-2 sentences reacting to the article.
Use the word "{slang}" naturally.
IMPORTANT: every number, name, and day must match the article exactly. Express anger through opinion and tone only.
Do NOT add any new facts or numbers. Language style: {language}.

Example:
{"analysis": "Facts: ₹2.1 crore, Monday, Nadi Bridge. Opinion: doubts they will finish on time.",
 "post": "Council passed ₹2.1 crore on Monday for Nadi Bridge. Let's see if they finish on time or if it's the usual tamasha."}

Return JSON: {"analysis": "...", "post": "..."}"""

PROMPT_D_MISINFO = """Article: {article}

This is a synthetic example of ONE distorted claim for detector training.
Think inside "analysis": (1) the original fact is {orig}, (2) the distorted fact to state is {fake}, (3) everything else stays as in the article.
Then write an angry {platform} post of 1-2 sentences using the distorted fact.
Use the word "{slang}" naturally.
{mutator_instruction}
Language style: {language}.

Example (original ₹2.1 crore, distorted ₹21 crore):
{"analysis": "Original: ₹2.1 crore. Distorted: ₹21 crore. Others unchanged: Monday, Nadi Bridge.",
 "post": "₹21 crore for Nadi Bridge repair?? Council passed it Monday. Total jumla, where is the rest going?"}

Return JSON: {"analysis": "...", "post": "..."}"""
