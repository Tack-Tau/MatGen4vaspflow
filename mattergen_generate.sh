#!/bin/bash
#SBATCH --job-name=mattergen_generate
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mattergen_gen_%j.out
#SBATCH --error=logs/mattergen_gen_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

# Load modules
module purge
module load cuda/11.8

# Activate conda environment
source ~/.bashrc
conda activate mattersim

# Navigate to working directory
cd $HOME/SOFT/mattergen_test/

# Create logs directory if it doesn't exist
mkdir -p logs

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Generation parameters
# Point to the run directory (contains config + checkpoints)
MODEL_PATH="outputs/singlerun/2025-10-08/10-40-35"
# MatterGen will automatically find the best checkpoint in checkpoints/ directory
OUTPUT_DIR="results/gen_electride_$(date +%Y%m%d_%H%M%S)"
NBATCHES=8
BATCH_SIZE=128
PROPERTY_VALUE=1  # 1 for electride, 0 for non-electride
GUIDANCE_FACTOR=3.0

echo "=========================================="
echo "MatterGen Conditional Generation"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batches: $NBATCHES * $BATCH_SIZE structures"
echo "Property: is_electride = $PROPERTY_VALUE"
echo "Guidance factor: $GUIDANCE_FACTOR"
echo "=========================================="

# Run generation
mattergen-generate $OUTPUT_DIR \
    --model_path=$MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --num_batches=$NBATCHES \
    --properties_to_condition_on='{"is_electride": '$PROPERTY_VALUE'}' \
    --diffusion_guidance_factor=$GUIDANCE_FACTOR \
    --trainer.accelerator=gpu \
    --trainer.devices=1 \
    --trainer.precision=32

echo "=========================================="
echo "Generation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# List generated files
ls -lh $OUTPUT_DIR/

