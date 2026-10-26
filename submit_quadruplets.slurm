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

# 1. Ensure UTF-8 locale on headless cluster nodes
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Load HPC Environment Modules
module load anaconda3-2025-12-2 2>/dev/null || true
module load cuda-12.4 2>/dev/null || module load cuda-12.1 2>/dev/null || true
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh 2>/dev/null || true

# 3. Activate Python Environment (e.g. pytorch)
conda activate pytorch 2>/dev/null || source activate pytorch 2>/dev/null || true

# 4. Check GPU status
nvidia-smi

# 5. Hugging Face Authentication Token for LLaMA-3.1 Gated Weights
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "⚠️ Warning: HF_TOKEN is not set in your environment."
    echo "   Meta-Llama-3.1-8B-Instruct is a gated model requiring Hugging Face authorization."
    echo "   Export your token via: export HF_TOKEN=\"hf_...\" before submitting or in your bashrc."
fi

# Set custom HuggingFace cache directory on fast cluster scratch storage if available
export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}

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
