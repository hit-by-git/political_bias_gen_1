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
  "split")
    echo "📊 Splitting dataset into event-level train/val/test..."
    python3 src/split_dataset.py
    ;;

  "task_a_muril")
    echo "🏋️ Training MuRIL on Task A (Post-Only)..."
    python3 src/train_encoder.py \
      --model_name_or_path google/muril-base-cased \
      --task post_only \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "task_b_muril")
    echo "🏋️ Training MuRIL on Task B (Evidence-Conditioned)..."
    python3 src/train_encoder.py \
      --model_name_or_path google/muril-base-cased \
      --task evidence_conditioned \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "task_b_indicbert")
    echo "🏋️ Training IndicBERT on Task B (Evidence-Conditioned)..."
    python3 src/train_encoder.py \
      --model_name_or_path ai4bharat/indic-bert \
      --task evidence_conditioned \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "task_b_xlmr")
    echo "🏋️ Training XLM-RoBERTa on Task B (Evidence-Conditioned)..."
    python3 src/train_encoder.py \
      --model_name_or_path xlm-roberta-base \
      --task evidence_conditioned \
      --epochs 3 \
      --batch_size 16 \
      $DEVICE_FP16
    ;;

  "llm_cot")
    echo "🤖 Evaluating Qwen2.5-14B with Chain-of-Thought on Test Split..."
    python3 src/eval_llm.py \
      --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 \
      --mode evidence_cot \
      --tp_size 2
    ;;

  "debias_cda")
    echo "🛡️ Running Contrastive Symmetrical Debiasing (CDA) on MuRIL..."
    python3 src/debiasing/contrastive_trainer.py \
      --model_name_or_path google/muril-base-cased \
      --lambda_reg 0.5 \
      --epochs 3 \
      --batch_size 8
    ;;

  "all_encoders")
    echo "🚀 Running all encoder baselines (MuRIL, IndicBERT, XLM-R)..."
    bash run_experiments.sh task_a_muril
    bash run_experiments.sh task_b_muril
    bash run_experiments.sh task_b_indicbert
    bash run_experiments.sh task_b_xlmr
    bash run_experiments.sh debias_cda
    ;;

  *)
    echo "Usage: bash run_experiments.sh [split|task_a_muril|task_b_muril|task_b_indicbert|task_b_xlmr|llm_cot|debias_cda|all_encoders]"
    exit 1
    ;;
esac

echo "=========================================================="
echo "✅ Finished: $TASK"
echo "Results stored in experiments/results/"
echo "=========================================================="
