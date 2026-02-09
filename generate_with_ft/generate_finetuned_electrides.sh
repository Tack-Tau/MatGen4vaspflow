#!/bin/bash
#SBATCH --job-name=gen_ternary_ft
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_finetuned_%j.out
#SBATCH --error=logs/gen_finetuned_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

# Load modules
module purge
module load cuda/11.8

# Activate conda environment
source ~/.bashrc
conda activate mattergen

# Navigate to working directory
cd $HOME/SOFT/mattergen_test/generate_with_ft

# Create logs directory if it doesn't exist
mkdir -p logs

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Generation parameters
CHECKPOINT="$HOME/SOFT/mattergen_test/outputs/singlerun/2025-10-08/10-40-35"
OUTPUT_DIR="../results/electrides_finetuned"
BATCH_SIZE=128
NUM_BATCHES=16
GUIDANCE_FACTOR=3.0
EXCESS_MIN=0.1
EXCESS_MAX=4.0

echo "=========================================="
echo "Ternary Electride Generation (Fine-tuned Model)"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $NUM_BATCHES"
echo "Total structures: $((BATCH_SIZE * NUM_BATCHES))"
echo "Guidance factor: $GUIDANCE_FACTOR"
echo "Excess electron range: $EXCESS_MIN - $EXCESS_MAX"
echo "=========================================="

# Run generation
python generate_with_finetuned.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_batches $NUM_BATCHES \
    --guidance_factor $GUIDANCE_FACTOR \
    --excess_min $EXCESS_MIN \
    --excess_max $EXCESS_MAX \
    --ternary_only

echo ""
echo "=========================================="
echo "Generation completed!"
echo "=========================================="

# Check results
if [ -f "$OUTPUT_DIR/electride_candidates.json" ]; then
    echo ""
    echo "Electride candidates found!"
    echo "Summary:"
    python -c "
import json
with open('$OUTPUT_DIR/electride_candidates.json', 'r') as f:
    candidates = json.load(f)
    total = sum(len(v['cif_files']) for v in candidates.values())
    print(f'  Total electride candidates: {total}')
    print(f'  Unique compositions: {len(candidates)}')
    print(f'')
    print('Top 10 compositions by excess electrons:')
    sorted_comps = sorted(candidates.items(), key=lambda x: x[1]['excess_electrons'], reverse=True)[:10]
    for i, (comp, data) in enumerate(sorted_comps, 1):
        n = len(data['cif_files'])
        e = data['excess_electrons']
        print(f'    {i}. {comp:15s} | {e:5.2f} e⁻ | {n:3d} structures')
"
fi

echo ""
echo "Results location: $OUTPUT_DIR"
echo "  - generated_crystals_cif.zip (all structures)"
echo "  - electride_candidates.json (filtered ternary electrides)"
echo "  - filtering_stats.json (statistics)"
echo ""
echo "Next steps:"
echo "  1. Review electride_candidates.json for compositions"
echo "  2. Extract structures with extract_composition_structures.py"
echo "  3. Evaluate stability and electride probability"
echo "=========================================="

