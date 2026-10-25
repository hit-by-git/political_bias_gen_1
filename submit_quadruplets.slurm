#!/bin/bash
# ==============================================================================
# Slurm Job Submission Script for Dhirubhai Ambani University Tyrone HPC Cluster
# Task: 2x2 Factorial Contrastive Quadruplet Synthesis (vLLM on Llama-3.1-8B)
# ==============================================================================
#SBATCH --job-name=quad_synthesis
#SBATCH --partition=h200-fullgpu          # Options: h200-fullgpu (141GB) or rtx6000-fullgpu (96GB)
#SBATCH --gres=gpu:h200:1                 # Use gpu:h200:1 for h200, or gpu:rtx_pro:1 for rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/quad_synthesis_%j.out
#SBATCH --error=logs/quad_synthesis_%j.err

echo "=========================================================="
echo "🎯 Job ID: $SLURM_JOB_ID running on $(hostname)"
echo "Started at: $(date)"
echo "=========================================================="

cd $SLURM_SUBMIT_DIR
mkdir -p logs output

# 1. Load HPC Environment Modules
module load anaconda3-2025-12-2
module load cuda-13.0
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh

# 2. Activate Python Environment (e.g. pytorch)
conda activate pytorch

# 3. Check GPU status
nvidia-smi

# 4. (Optional) Set your HuggingFace Token if using gated models like meta-llama
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"

# Model selection (default Meta Llama 3.1 8B Instruct, or fallback to Qwen 2.5 14B)
MODEL_NAME=${SYNTHESIS_MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}

echo "🚀 Launching vLLM Quadruplet Synthesis with Model: $MODEL_NAME"
python3 src/synthesis/pipeline.py \
    --format quadruplet \
    --model "$MODEL_NAME" \
    --tp 1 \
    --resume

echo "=========================================================="
echo "✅ Job Completed at: $(date)"
echo "Results written to output/synthetic_quadruplets.jsonl"
echo "=========================================================="
