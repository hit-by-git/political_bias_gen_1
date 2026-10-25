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
- **Dual-Stage Blacklist Validation:** Articles and posts are checked against an expanded Indian political gazetteer (`data/processed/real_names.txt`). Any instance of real entity leakage triggers immediate rejection.

---

## 📁 Repository Structure

```text
political_bias_gen_1/
├── docs/                                     # Master Thesis & Research Deliverables
│   ├── RESEARCH_REPORT.md                    # Comprehensive Research Project Report
│   └── thesis_report.tex                     # Master LaTeX Thesis Document
│
├── data/                                     # Data Pathways
│   ├── raw/                                  # Raw external event data (GDELT 1,500 scrape)
│   │   └── gdelt_india_political_1500.csv
│   ├── processed/                            # Enriched headlines & entity gazetteer
│   │   ├── gdelt_india_political_1500_enriched.csv
│   │   └── real_names.txt
│   └── splits/                               # Event-stratified splits (train/val/test)
│       ├── train.jsonl                       # 2,304 posts (768 events)
│       ├── val.jsonl                         # 495 posts (165 events)
│       ├── test.jsonl                        # 489 posts (163 events)
│       └── split_summary.json
│
├── output/                                   # Synthetic Triplet Dataset Outputs
│   ├── synthetic_dataset.jsonl               # 3,288 verified contrastive triplets
│   ├── rejections.jsonl                      # 402 rejected events log
│   └── synthetic_thesis_dataset.zip          # Distribution zip archive
│
├── src/                                      # Modular Code Pathways
│   ├── synthesis/                            # 🚀 IMPLEMENTATION 1: Data Generation Engine
│   │   ├── pipeline.py                       # vLLM high-throughput generation driver
│   │   ├── pull_gdelt.py                     # GDELT 2.0 event harvester
│   │   ├── enrich_headlines.py               # Slug & OpenGraph metadata enrichment
│   │   ├── prompts.py                        # Multilingual CoT few-shot prompt templates
│   │   ├── mutators.py                       # 5 factual perturbation operators
│   │   ├── validators.py                     # Script-aware, slang & entity validators
│   │   └── sitecustomize.py                  # Colab/Kaggle CUDA isolation patch
│   │
│   └── evaluation/                           # 🏋️ IMPLEMENTATION 2: Benchmarking & Debiasing Engine
│       ├── split_dataset.py                  # Stratified event-level dataset partitioner
│       ├── train_encoder.py                  # Task A (Post-Only) & Task B (Evidence) trainer
│       ├── eval_llm.py                       # LLM Zero-Shot & CoT evaluation
│       ├── evaluate_metrics.py               # Diagnostic metrics (ΔABG, TCR, ΔCL, Macro-F1)
│       └── debiasing/
│           └── contrastive_trainer.py        # CDA symmetrical contrastive regularization
│
└── experiments/                              # Empirical Benchmarks & Deliverables
    ├── run_experiments.sh                    # Master runner script for all experiments
    ├── checkpoints/                          # Model checkpoints (.gitkeep)
    └── results/                              # 9 Empirical evaluation deliverables
        ├── debiased_muril_metrics.json
        ├── debiased_muril_table.tex
        ├── debiased_muril_test_preds.jsonl
        ├── google_muril-base-cased_evidence_conditioned_metrics.json
        ├── google_muril-base-cased_evidence_conditioned_table.tex
        ├── google_muril-base-cased_evidence_conditioned_test_preds.jsonl
        ├── google_muril-base-cased_post_only_metrics.json
        ├── google_muril-base-cased_post_only_table.tex
        └── google_muril-base-cased_post_only_test_preds.jsonl
```

---

## 🚀 Execution & Usage

### 1. Requirements
```bash
pip install -r requirements.txt
# For GPU acceleration (Kaggle / Google Colab 2x T4 or A100):
pip install vllm
```

### 2. Implementation 1: Data Synthesis Pipeline
```bash
# Dry run / mock testing (CPU / Mac):
python src/synthesis/pipeline.py --limit 10 --mock

# Production generation (GPU / vLLM):
python src/synthesis/pipeline.py --tp 2 --resume

# Optional: Harvest fresh GDELT articles & enrich headlines:
python src/synthesis/pull_gdelt.py --count 100
python src/synthesis/enrich_headlines.py
```

### 3. Implementation 2: Downstream Benchmarking & Debiasing
```bash
# Partition synthetic dataset into event-stratified train/val/test splits:
bash run_experiments.sh split

# Train MuRIL on Task A (Post-Only bias probing):
bash run_experiments.sh task_a_muril

# Train MuRIL on Task B (Evidence-Conditioned verification):
bash run_experiments.sh task_b_muril

# Run Symmetrical Contrastive Debiasing (CDA):
bash run_experiments.sh debias_cda

# Run all encoder baselines (MuRIL, IndicBERT, XLM-R, CDA):
bash run_experiments.sh all_encoders
```

---

## 📊 Dataset Schema (`output/synthetic_dataset.jsonl`)
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
