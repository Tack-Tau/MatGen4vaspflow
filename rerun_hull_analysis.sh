#!/bin/bash
#SBATCH --job-name=hull_analysis
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/hull_analysis_%j.out
#SBATCH --error=logs/hull_analysis_%j.err
#SBATCH --mail-type=END,FAIL
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

# Set number of threads for CPU parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Evaluation parameters
RESULTS_PATH="results/gen_electride_20251009_205357/generated_crystals_cif.zip"
OUTPUT_DIR="$(dirname $RESULTS_PATH)"

echo "=========================================="
echo "Rerunning Hull Analysis"
echo "=========================================="
echo "Input structures: $RESULTS_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "ENERGY REFERENCE: MatterSim"
echo "  - Your structures: Already relaxed with MatterSim + FIRE"
echo "  - MP competing phases: Will be re-relaxed with MatterSim + FIRE"
echo "  - Result: Consistent energy scale for valid hull analysis"
echo ""
echo "Note: MP structures cached for fast subsequent runs"
echo "=========================================="
echo ""

# Check MP_API_KEY
if [ -z "$MP_API_KEY" ]; then
    echo "ERROR: MP_API_KEY not set!"
    echo "  Run: export MP_API_KEY='your_key'"
    echo "  Get key at: https://next-gen.materialsproject.org/api"
    exit 1
fi

echo "MP_API_KEY: ${MP_API_KEY:0:10}..."
echo ""

# Run hull analysis (will auto-detect relaxed_structures.extxyz)
# MP structures will be re-relaxed with MatterSim for consistent energy reference
python evaluate_stability_hull.py \
    --stability-json "$OUTPUT_DIR/stability.json" \
    --mp-api-key "$MP_API_KEY" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --plot-diagrams

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS!"
    echo "=========================================="
    echo "  Results: $OUTPUT_DIR/hull_stability.json"
    echo "  MP cache: $OUTPUT_DIR/mp_mattersim_cache/"
    echo "  Phase diagrams: $OUTPUT_DIR/phase_diagrams/"
    echo ""

    # Show summary
    echo "Quick summary:"
    python -c "
import json
with open('$OUTPUT_DIR/hull_stability.json', 'r') as f:
    data = json.load(f)
    stable = [d for d in data if d.get('is_stable', False)]
    print(f'  Total structures: {len(data)}')
    print(f'  Stable (E_hull < 0.01 eV/atom): {len(stable)} ({100*len(stable)/len(data):.1f}%)')

    if stable:
        print(f'\\n  Most stable structures:')
        sorted_stable = sorted(stable, key=lambda x: x['energy_above_hull'])
        for s in sorted_stable[:10]:
            print(f\"    • {s['formula']}: E_hull = {s['energy_above_hull']:.6f} eV/atom\")
"
    echo ""
    echo "=========================================="
    echo "Job completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "ERROR: Hull analysis failed!"
    echo "Check the error messages above."
    exit 1
fi

