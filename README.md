# Multilingual Synthetic Misinformation Benchmark Pipeline

This pipeline turns real-world GDELT political event signals into a strictly controlled, **multilingual counterfactual benchmark** for training and evaluating misinformation detectors and auditing affective bias in NLP models.

It uses `vLLM` with the `Qwen/Qwen2.5-14B-Instruct-AWQ` model to generate **balanced contrastive triplets** across three linguistic registers:
1. **Indian English** (Plain & Casual)
2. **Authentic Romanized Hinglish** (Colloquial syntactic code-mixing)
3. **Devanagari Hindi (मानक हिंदी)**

---

## 🎯 Core Research Motive
Standard NLP classifiers frequently suffer from an **Affective Bias Confounder**: they learn to associate angry, sarcastic, or vernacular slang (*jumla*, *dhokha*, *tamasha*, *ghotala*, *babu*) with "misinformation," failing to recognize angry commentary that is 100% factually accurate.

To eliminate this shortcut, this pipeline enforces **Strict Triplet Atomicity**: each event generates three strictly matched variants:
- **Neutral:** Calm, factual summary.
- **Hard Negative:** Angry, sarcastic, slang-laden reaction containing **100% accurate facts**.
- **Misinformation:** Angry, sarcastic, slang-laden post containing **exactly ONE mutated fact** (distorted amount, count, date, outcome, or actor attribution).

Because every accepted event produces all three variants ($N_{neutral} = N_{hard\_neg} = N_{misinfo}$), a detector cannot achieve high performance through tone or keyword heuristics alone.

---

## 🔒 Safety & Ethical Safeguards
- **100% Synthetic Grounding:** All named actors are strictly fictional (e.g., *Mayor Verma*, *Candidate Rajesh*, *Party Vikas*).
- **Pre-Sanitization:** Real party names (BJP, Congress, TMC, etc.) and political figures are scrubbed from raw GDELT headlines before prompt injection.
- **Dual-Stage Blacklist Validation:** Articles and posts are checked against an expanded Indian political gazetteer (`real_names.txt`). Any instance of real entity leakage triggers immediate rejection.

---

## 🛠️ Pipeline Architecture
1. **Headline Pre-Sanitization:** Scrubs real political entities from GDELT headlines.
2. **Stage 1 (Article Expansion):** Generates a 3-sentence factual anchor containing a rupee amount, crowd count, and weekday.
3. **Multilingual Fact Extraction & Mutation:** Extracts atomic facts with bilingual regexes and applies non-linear perturbations (e.g. non-round multipliers, outcome polarity inversion).
4. **Stage 2 (Symmetrical CoT Generation):** Generates `neutral`, `hard_negative`, and `misinfo` posts using explicit in-context few-shot code-mixing and Hindi guidelines.
5. **Script-Aware Dual-Stage Validation:** Normalizes Devanagari numerals and validates fact retention, slang presence, and number integrity.
6. **Atomic Persistence:** Only writes to `synthetic_dataset.jsonl` if **all three variants** pass validation.

---

## 🚀 Usage

### 1. Requirements
```bash
pip install -r requirements.txt
# For GPU acceleration (Kaggle / Google Colab 2x T4 or A100):
pip install vllm
```

### 2. Local Testing / Dry-Run (CPU / Mac)
To verify the pipeline logic without a GPU:
```bash
python pipeline.py --limit 10 --mock
```

### 3. Production Run (GPU / vLLM)
On Google Colab or Kaggle (2x T4 GPUs):
```bash
python pipeline.py --tp 2 --resume
```

---

## 📊 Dataset Schema (`synthetic_dataset.jsonl`)
Each line is a JSON object with:
- `id`: Unique identifier (`<event_id>_<variant>`)
- `event_id`: GDELT root event ID
- `article`: Synthetic ground-truth 3-sentence article
- `post`: Generated social media post
- `label`: `"not_misinfo"` (neutral & hard_negative) vs. `"misinfo"`
- `variant`: `"neutral"`, `"hard_negative"`, or `"misinfo"`
- `tactic`: Mutation tactic (`fabricated_amount`, `inflated_count`, `wrong_date`, `reversed_outcome`, `false_attribution`, or `none`)
- `contradicting_fact`: Original ground-truth fact
- `false_claim`: Distorted counterfactual claim
- `platform`: `X post`, `WhatsApp forward`, or `Facebook comment`
- `language`: `plain English`, `Hinglish (Roman script)`, or `Devanagari Hindi`
- `slang`: Required colloquial slang term (in Latin or Devanagari)
- `analysis`: Chain-of-thought fact preservation rationale
