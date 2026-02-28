#!/bin/bash
#SBATCH --job-name=internvl_multi_gpu
#SBATCH --partition=kamiak
#SBATCH --gres=gpu:tesla:2              
#SBATCH --constraint=h100               
#SBATCH --cpus-per-task=32              
#SBATCH --mem=128G                      
#SBATCH --time=24:00:00
#SBATCH --output=logs/internvl_%j.out
#SBATCH --error=logs/internvl_%j.err

mkdir -p logs

# Load modules
module load cuda/12.2.0
module load cudnn/8.9.7_cuda12.2

# Set cache
export HF_HOME=/data/lab/lapin/UVSD_Enrichment/.cache
export HUGGINGFACE_HUB_CACHE=/data/lab/lapin/UVSD_Enrichment/.cache

# GPU info
echo "========== GPU INFO =========="
nvidia-smi
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=============================="

# Run with full python path (no activation needed)
~/.conda/envs/internvl_env/bin/python internvl_inference.py
