#!/usr/bin/env bash
# ==============================================================================
# run_experiments.sh
# Master execution runner for Phase 2 Thesis Experiments:
# 1. Post-Only Bias Probing (Task A) on MuRIL, IndicBERT, XLM-R
# 2. Evidence-Conditioned Claim Verification (Task B) on MuRIL, IndicBERT, XLM-R
# 3. LLM Zero-Shot & CoT Evaluation on Qwen2.5-14B / LLaMA-3.1-8B
# 4. Symmetrical Contrastive Debiasing (CDA)
# ==============================================================================

set -e

# Default settings
TASK=${1:-"all"}
DEVICE_FP16=${2:-"--fp16"}

echo "=========================================================="
echo "🎯 STARTING THESIS EXPERIMENTAL BENCHMARK SUITE"
echo "Target Task: $TASK"
echo "=========================================================="

mkdir -p experiments/checkpoints experiments/results

case $TASK in
  "synthesis_quadruplets")
    echo "🚀 Running 2x2 Factorial Quadruplet Synthesis (vLLM)..."
    python3 src/synthesis/pipeline.py \
      --format quadruplet \
      --model ${SYNTHESIS_MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"} \
      --tp 1 \
      --resume
    ;;

  "split")
    echo "📊 Splitting dataset into event-level train/val/test..."
    python3 src/evaluation/split_dataset.py
    ;;

  "task_a_muril")
    echo "🏋️ Training MuRIL on Task A (Post-Only)..."
    python3 src/evaluation/train_encoder.py \
      --model_name_or_path google/muril-base-cased \
      --task post_only \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "task_b_muril")
    echo "🏋️ Training MuRIL on Task B (Evidence-Conditioned)..."
    python3 src/evaluation/train_encoder.py \
      --model_name_or_path google/muril-base-cased \
      --task evidence_conditioned \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "task_b_indicbert")
    echo "🏋️ Training IndicBERT on Task B (Evidence-Conditioned)..."
    python3 src/evaluation/train_encoder.py \
      --model_name_or_path ai4bharat/indic-bert \
      --task evidence_conditioned \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "task_b_xlmr")
    echo "🏋️ Training XLM-RoBERTa on Task B (Evidence-Conditioned)..."
    python3 src/evaluation/train_encoder.py \
      --model_name_or_path xlm-roberta-base \
      --task evidence_conditioned \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "llm_cot")
    echo "🤖 Evaluating LLaMA-3.1-8B with Chain-of-Thought on Test Split..."
    python3 src/evaluation/eval_llm.py \
      --model ${SYNTHESIS_MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"} \
      --mode evidence_cot \
      --tp_size 1
    ;;

  "debias_cda")
    echo "🛡️ Running Contrastive Symmetrical Debiasing (CDA) on MuRIL..."
    python3 src/evaluation/debiasing/contrastive_trainer.py \
      --model_name_or_path google/muril-base-cased \
      --lambda_reg 0.5 \
      --epochs 3 \
      --batch_size 8
    ;;

  "eval_step1_test")
    echo "🔍 Step 1: Evaluating on Held-Out Contrastive Test Split (Causal Diagnostic)..."
    python3 src/evaluation/evaluate_metrics.py \
      --predictions experiments/results/debiased_muril_test_preds.jsonl \
      --model_name "MuRIL + CDA (Ours)" \
      --latex_out experiments/results/step1_test_latex.tex
    ;;

  "eval_step2_hasoc")
    echo "🛡️ Step 2: Evaluating on HASOC 2019 (Toxicity Confounding Audit)..."
    python3 src/evaluation/eval_hasoc.py \
      --model_path experiments/checkpoints/contrastive_muril \
      --hasoc_test_file data/HASOC_eng/hasoc2019_en_test-2919.tsv
    ;;

  "eval_step3_constraint")
    echo "🌐 Step 3: Evaluating on Constraint@AAAI 2021 (Real-World Social Media)..."
    python3 src/evaluation/eval_constraint.py \
      --model_path experiments/checkpoints/contrastive_muril \
      --test_file data/constraint_aaai2021/test.csv \
      $DEVICE_FP16
    ;;

  "eval_all_3steps")
    echo "🎯 Running Full 3-Step Evaluation Protocol..."
    bash run_experiments.sh eval_step1_test
    bash run_experiments.sh eval_step2_hasoc
    bash run_experiments.sh eval_step3_constraint
    ;;

  "all_encoders")
    echo "🚀 Running all encoder baselines (MuRIL, IndicBERT, XLM-R, CDA)..."
    bash run_experiments.sh task_a_muril
    bash run_experiments.sh task_b_muril
    bash run_experiments.sh task_b_indicbert
    bash run_experiments.sh task_b_xlmr
    bash run_experiments.sh debias_cda
    bash run_experiments.sh eval_all_3steps
    ;;

  *)
    echo "Usage: bash run_experiments.sh [synthesis_quadruplets|split|task_a_muril|task_b_muril|task_b_indicbert|task_b_xlmr|llm_cot|debias_cda|eval_step1_test|eval_step2_hasoc|eval_step3_constraint|eval_all_3steps|all_encoders]"
    exit 1
    ;;
esac

echo "=========================================================="
echo "✅ Finished: $TASK"
echo "Results stored in experiments/results/"
echo "=========================================================="
