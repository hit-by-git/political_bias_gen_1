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
mkdir -p logs experiments/checkpoints experiments/results data/splits

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
pip install -q -U "regex>=2025.10.22" 2>/dev/null || pip install -q -U --user "regex>=2025.10.22" 2>/dev/null || true

# 4. Check GPU status
nvidia-smi

# 5. Run complete encoder suite and 3-step evaluation protocol
echo "🚀 Launching all encoder baselines & 3-step evaluation protocol..."
bash run_experiments.sh all_encoders

echo "=========================================================="
echo "✅ Job Completed at: $(date)"
echo "Results written to experiments/results/"
echo "=========================================================="
