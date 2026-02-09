#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

echo "=========================================="
echo "GPU and CUDA Test Script (SLURM Job)"
echo "=========================================="

# Load CUDA module
echo "Loading CUDA module..."
module load cuda/11.8

# Activate conda environment
echo "Activating mattergen environment..."
source ~/.bashrc
conda activate mattergen

# Check CUDA module loaded
echo -e "\n--- CUDA Module Check ---"
module list
echo "CUDA path: $(which nvcc)"
nvcc --version | grep "release"

# Check GPU availability
echo -e "\n--- GPU Check ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Test PyTorch CUDA
echo -e "\n--- PyTorch CUDA Test ---"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# Test torch_scatter
echo -e "\n--- Testing torch_scatter ---"
python -c "
import torch
try:
    import torch_scatter
    print('torch_scatter imported successfully')
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f'CUDA tensor created: {x}')
        print(' torch_scatter with CUDA: OK!')
    else:
        print('  CUDA not available, torch_scatter test skipped')
except Exception as e:
    print(f'  torch_scatter error: {e}')
    print('Need to reinstall with: pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html')
"

echo -e "\n=========================================="
echo "Test completed!"
echo "=========================================="

