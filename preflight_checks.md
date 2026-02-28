Pre-Flight Checklist for SLURM Batch Jobs
1. Environment Setup
Check conda/virtual environment exists and has packages:

bash
# List environments
conda env list
ls -la ~/.conda/envs/

# Check what's installed
~/.conda/envs/YOUR_ENV/bin/pip list

# Test imports
~/.conda/envs/YOUR_ENV/bin/python -c "import torch, transformers, pandas"
Verify Python version:

bash
~/.conda/envs/YOUR_ENV/bin/python --version
2. GPU Resources
Check available GPU types and availability:

bash
# See all partitions with GPU info
sinfo -o "%20P %5a %.10l %16F %8G"

# Check specific GPU types
sinfo -p kamiak -o "%P %G %C %m %N"

# See what GPU types are available
scontrol show partition kamiak | grep -i gres

# Check current GPU usage
squeue -p kamiak -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %b"
Check GPU constraints and types:

bash
# See nodes with specific GPU types
sinfo -N -o "%N %G %C %m" | grep h100
sinfo -N -o "%N %G %C %m" | grep a100

# Check how many GPUs per node
scontrol show node sn16 | grep -i gres
Verify your script's GPU request is valid:

bash
# Your request format should match available resources
#SBATCH --gres=gpu:tesla:2       # Request 2 Tesla GPUs
#SBATCH --constraint=h100        # Specifically H100s

# Common formats:
# --gres=gpu:1              # Any 1 GPU
# --gres=gpu:2              # Any 2 GPUs
# --gres=gpu:tesla:2        # 2 Tesla GPUs (brand)
# --gres=gpu:h100:2         # 2 H100 GPUs (specific model)
Test GPU access in interactive job:

bash
# Request interactive session with GPUs
salloc --partition=kamiak --gres=gpu:tesla:2 --constraint=h100 --time=1:00:00

# Once allocated, check GPUs
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Test PyTorch can see them
~/.conda/envs/YOUR_ENV/bin/python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); print(torch.cuda.get_device_name(0))"

# Exit when done
exit
Monitor GPU usage during job:

bash
# Find which node your job is on
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

# SSH to that node (e.g., sn16)
ssh sn16

# Watch GPU usage
watch -n 1 nvidia-smi

# Or check once
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Exit back to login node
exit
Check GPU memory requirements:

bash
# Estimate model memory (rough calculation)
# Float32: 4 bytes per parameter
# Float16: 2 bytes per parameter
# BFloat16: 2 bytes per parameter

# Example: 26B parameter model in BFloat16
# ~26B * 2 bytes = 52 GB minimum
# Add ~20-30% for activations/gradients = ~70 GB total

# Check GPU memory available
sinfo -N -o "%N %G" | grep h100
# H100 PCIe = ~80 GB
# A100 = 40 GB or 80 GB depending on variant
3. Module Compatibility
Check available modules:

bash
module avail cuda
module avail cudnn
module avail anaconda
Verify CUDA version matches PyTorch:

bash
# Check PyTorch CUDA version
~/.conda/envs/YOUR_ENV/bin/python -c "import torch; print(torch.version.cuda)"

# This should match the module you're loading
# e.g., if torch needs CUDA 12.1, load cuda/12.2.0 (compatible)
4. File Paths and Directories
Create required directories:

bash
mkdir -p logs
mkdir -p output
mkdir -p checkpoints
Verify input files exist:

bash
ls -lh your_dataset.json
ls -lh /path/to/images/
Check file permissions:

bash
ls -la internvl_inference.py  # Should be readable
5. SLURM Script Validation
Check syntax:

bash
bash -n run_internvl.sbatch  # Checks for syntax errors
Verify #SBATCH directives:

Output/error paths point to existing directories

Partition name is correct (sinfo to see available partitions)

GPU type exists (sinfo -o "%P %G" for GPU info)

Memory/CPU requests are reasonable

Time limit is appropriate

Test key commands interactively first:

bash
# Load modules
module load cuda/12.2.0
module load cudnn/8.9.7_cuda12.2

# Test activation/python
~/.conda/envs/YOUR_ENV/bin/python --version

# Test imports
~/.conda/envs/YOUR_ENV/bin/python -c "import torch; print(torch.cuda.is_available())"
6. Resource Requirements
Check available resources:

bash
# See available partitions and their limits
sinfo

# Check GPU availability
sinfo -o "%P %G %C %m"

# See current queue
squeue -p kamiak
Estimate requirements:

Memory: Check model size + data loading needs

GPUs: How many does your script actually use?

Time: Run small test first to estimate

7. Cache and Environment Variables
Set up cache directories:

bash
export HF_HOME=/path/to/.cache
export HUGGINGFACE_HUB_CACHE=/path/to/.cache
export TORCH_HOME=/path/to/.cache

# Make sure they exist
mkdir -p $HF_HOME
Check disk space:

bash
df -h /path/to/output/
df -h /path/to/cache/
8. Test Run
Do a dry run or small test:

bash
# Interactive job first
salloc --partition=kamiak --gres=gpu:1 --time=1:00:00

# Once allocated, test manually:
module load cuda/12.2.0
~/.conda/envs/YOUR_ENV/bin/python internvl_inference.py --test_mode
9. Monitoring Setup
Prepare monitoring commands:

bash
# Save these in a file for easy access
cat > check_job.sh << 'EOF'
#!/bin/bash
JOBID=$1
echo "=== Job Status ==="
squeue -j $JOBID
echo -e "\n=== Recent Output ==="
tail -20 logs/internvl_${JOBID}.out
echo -e "\n=== Any Errors? ==="
tail -20 logs/internvl_${JOBID}.err
EOF

chmod +x check_job.sh
Quick Pre-Submission Checklist
bash
# ✓ Environment exists and has packages
~/.conda/envs/YOUR_ENV/bin/pip list | grep torch

# ✓ GPU resources are available
sinfo -p kamiak -o "%P %G %a"

# ✓ GPU request matches available hardware
sinfo -N -o "%N %G" | grep h100

# ✓ PyTorch can access GPUs
~/.conda/envs/YOUR_ENV/bin/python -c "import torch; print(torch.cuda.is_available())"

# ✓ CUDA modules match PyTorch version
~/.conda/envs/YOUR_ENV/bin/python -c "import torch; print(torch.version.cuda)"

# ✓ Logs directory exists
mkdir -p logs

# ✓ Script syntax is valid
bash -n run_internvl.sbatch

# ✓ Python path is correct
which python  # Should NOT be /usr/bin/python in your script

# ✓ Test imports work
~/.conda/envs/YOUR_ENV/bin/python -c "import torch, transformers, pandas"

# ✓ Input files exist
ls -lh your_data_file.json

# ✓ Cache directories exist
ls -ld $HF_HOME

# ✓ Ready to submit!
sbatch run_internvl.sbatch
GPU-Specific Monitoring Commands
After job starts:

bash
# Get node name
squeue -u $USER -o "%.18i %.9P %.8u %R"

# SSH and monitor
ssh NODE_NAME
watch -n 1 nvidia-smi

# Check GPU utilization percentage
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Full stats
nvidia-smi dmon -s pucvmet
Common Mistakes We Encountered
❌ Assuming conda environment has packages (it was empty!)

❌ Using conda activate without loading anaconda module

❌ CUDA version mismatch (11.8 vs 12.2)

❌ Output directory doesn't exist

❌ Using system python instead of environment python

❌ Not checking activate script exists before sourcing it

❌ Not verifying GPU type/availability before submission

❌ Requesting more GPUs than available per node

Following this checklist would have caught all our issues before submission!