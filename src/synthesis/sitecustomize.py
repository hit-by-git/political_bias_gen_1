# sitecustomize.py
# Automatically executed by Python on startup for both the main process and all vLLM subprocesses.
import sys

# Mask torchaudio to prevent CUDA 13 vs CUDA 12.8 mismatch crash on Kaggle / Colab
sys.modules['torchaudio'] = None
