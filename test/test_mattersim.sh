#!/bin/bash
#SBATCH --job-name=test_mattersim
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_mattersim_%j.out
#SBATCH --error=logs/test_mattersim_%j.err

# Load modules
module purge
module load cuda/11.8

# Activate conda environment
source ~/.bashrc
conda activate mattersim

# Navigate to working directory
cd $HOME/SOFT/mattersim_test/

# Create directories
mkdir -p logs

# Run relaxation test
echo ""
echo "=========================================="
echo "Testing MatterSim Relaxation"
echo "=========================================="

python test_mattersim_relax.py gen_0.cif

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

# Show output files
echo ""
echo "Output files:"
ls -lh test_relaxed_*.cif 2>/dev/null || echo "No relaxed structures saved (check errors above)"

