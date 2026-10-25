#!/bin/bash
# ==============================================================================
# Slurm Job Submission Script for Dhirubhai Ambani University Tyrone HPC Cluster
# Task: Encoder Benchmarking Suite (MuRIL, IndicBERT, XLM-R & Contrastive CDA)
# ==============================================================================
#SBATCH --job-name=encoder_suite
#SBATCH --partition=rtx6000-fullgpu       # RTX 6000 Blackwell (96GB) or h200-fullgpu (141GB)
#SBATCH --gres=gpu:rtx_pro:1              # Use gpu:rtx_pro:1 for rtx6000 or gpu:h200:1 for h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/encoder_suite_%j.out
#SBATCH --error=logs/encoder_suite_%j.err

echo "=========================================================="
echo "🎯 Job ID: $SLURM_JOB_ID running on $(hostname)"
echo "Started at: $(date)"
echo "=========================================================="

cd $SLURM_SUBMIT_DIR
mkdir -p logs experiments/checkpoints experiments/results

# 1. Load HPC Environment Modules
module load anaconda3-2025-12-2
module load cuda-13.0
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh

# 2. Activate Python Environment (e.g. pytorch)
conda activate pytorch

# 3. Check GPU status
nvidia-smi

# 4. Run all encoders and debiasing benchmark
bash run_experiments.sh all_encoders

echo "=========================================================="
echo "✅ Job Completed at: $(date)"
echo "Results written to experiments/results/"
echo "=========================================================="
