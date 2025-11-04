#!/bin/bash
#SBATCH --job-name=mattergen_eval
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mattergen_eval_%j.out
#SBATCH --error=logs/mattergen_eval_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
module load cuda/11.8

# Activate conda environment
source ~/.bashrc
conda activate mattersim

# Navigate to working directory
cd $HOME/SOFT/mattergen_test/

# Create logs directory if it doesn't exist
mkdir -p logs

# Verify MP_API_KEY is available
if [ -z "$MP_API_KEY" ]; then
    echo "WARNING: MP_API_KEY not found in environment"
    echo "  Hull analysis will be skipped"
else
    echo "MP_API_KEY found: ${MP_API_KEY:0:10}... (using for hull analysis)"
fi

# Set number of threads for CPU parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Evaluation parameters
# UPDATE THIS PATH to your generated results
RESULTS_PATH="results/gen_electride_20251009_205357/generated_crystals_cif.zip"
OUTPUT_DIR="$(dirname $RESULTS_PATH)"
MATTERSIM_POTENTIAL="MatterSim-v1.0.0-5M.pth"  # Adjust path if needed
CLASSIFIER_MODEL_DIR="models/electride_classifier"

echo "=========================================="
echo "MatterGen Structure Evaluation"
echo "=========================================="
echo "Input structures: $RESULTS_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "MatterSim potential: $MATTERSIM_POTENTIAL"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Step 1: Manual relaxation (workaround for BatchRelaxer API issue)
echo "Step 1: Relaxing structures manually..."
echo "Note: Using manual relaxation with checkpointing (can resume if interrupted)"
echo ""

python relax_structures_manual.py \
    --structures_zip "$RESULTS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --fmax 0.001 \
    --max_steps 500
    # Add --max_structures 100 to limit processing if needed

if [ $? -eq 0 ]; then
    echo ""
    echo "  Manual relaxation completed!"
    echo "  Relaxed structures: $OUTPUT_DIR/relaxed_structures.extxyz"
    echo ""
else
    echo ""
    echo "  WARNING: Relaxation failed or was interrupted"
    echo "  Checkpoint saved - rerun this script to resume"
    echo ""
    exit 1
fi

# Step 2: Run mattergen-evaluate (OPTIONAL - skip if fails)
echo "=========================================="
echo "Step 2: Computing validity, novelty metrics (optional)..."
echo "=========================================="

# Note: This step often fails due to corrupted MP2020 reference data in MatterGen
# It's not critical for electride screening, so we skip if it fails
if mattergen-evaluate "$RESULTS_PATH" \
    --relax=False \
    --potential_load_path="$MATTERSIM_POTENTIAL" \
    --save_as="$OUTPUT_DIR/mattergen_metrics.json" \
    --device=cuda 2>&1 | tee /tmp/mattergen_eval.log; then
    echo ""
    echo "  Metrics saved to: $OUTPUT_DIR/mattergen_metrics.json"
else
    echo ""
    echo "  WARNING: mattergen-evaluate failed (known issue with reference data)"
    echo "  Skipping this step - not critical for stability analysis"
fi

echo ""
echo "=========================================="
echo "Step 3: Computing energy above hull (thermodynamic stability)..."
echo "=========================================="
echo ""
echo "ENERGY REFERENCE: MatterSim"
echo "  - Your structures: Already relaxed with MatterSim"
echo "  - MP competing phases: Will be re-relaxed with MatterSim"
echo "  - Result: Consistent energy scale for valid hull analysis"
echo ""
echo "Note: This is fast screening. For publication, re-run with VASP energies."
echo ""

if [ -z "$MP_API_KEY" ]; then
    echo ""
    echo "WARNING: MP_API_KEY not set - skipping hull analysis"
    echo "  Set it with: export MP_API_KEY='your_key'"
    echo "  Get your key at: https://next-gen.materialsproject.org/api"
    echo ""
else
    # Use relaxed structures (auto-detected from OUTPUT_DIR)
    # MP structures will be re-relaxed with MatterSim for consistent energy reference
    python evaluate_stability_hull.py \
        --stability-json "$OUTPUT_DIR/stability.json" \
        --mp-api-key "$MP_API_KEY" \
        --output-dir "$OUTPUT_DIR" \
        --device cuda \
        --plot-diagrams
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "  Hull analysis completed (MatterSim energy scale)!"
        echo "  Results: $OUTPUT_DIR/hull_stability.json"
        echo "  MP cache: $OUTPUT_DIR/mp_mattersim_cache/"
    else
        echo ""
        echo "  Hull analysis failed (check MP_API_KEY)"
    fi
fi

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "    $OUTPUT_DIR/stability.json"
echo "    $OUTPUT_DIR/relaxed_structures.extxyz"

if [ -f "$OUTPUT_DIR/hull_stability.json" ]; then
    echo "    $OUTPUT_DIR/hull_stability.json (MOST IMPORTANT)"
else
    echo "    $OUTPUT_DIR/hull_stability.json (failed - check MP_API_KEY)"
fi

if [ -f "$OUTPUT_DIR/mattergen_metrics.json" ]; then
    echo "    $OUTPUT_DIR/mattergen_metrics.json"
else
    echo "    $OUTPUT_DIR/mattergen_metrics.json (skipped - not critical)"
fi

echo ""
echo "=========================================="

# Show summary statistics
echo ""
echo "=========================================="
echo "SUMMARY STATISTICS"
echo "=========================================="

# Stability statistics (from manual relaxation)
if [ -f "$OUTPUT_DIR/stability.json" ]; then
    echo ""
    echo "Force-based relaxation:"
    python -c "
import json
with open('$OUTPUT_DIR/stability.json', 'r') as f:
    data = json.load(f)
    converged = [d for d in data if d.get('converged', False)]
    print(f'  Total structures: {len(data)}')
    print(f'  Converged: {len(converged)} ({100*len(converged)/len(data):.1f}%)')
    if converged:
        energies = [s['energy_per_atom'] for s in converged]
        print(f'  Mean energy/atom: {sum(energies)/len(energies):.4f} eV/atom')
        print(f'  Min energy/atom: {min(energies):.4f} eV/atom')
        print(f'  Max energy/atom: {max(energies):.4f} eV/atom')
"
fi

# Hull-based stability
if [ -f "$OUTPUT_DIR/hull_stability.json" ]; then
    echo ""
    echo "Thermodynamic stability (convex hull):"
    python -c "
import json
with open('$OUTPUT_DIR/hull_stability.json', 'r') as f:
    data = json.load(f)
    stable = [d for d in data if d.get('is_stable', False)]
    print(f'  Total analyzed: {len(data)}')
    print(f'  Stable (E_hull < 0.01 eV/atom): {len(stable)} ({100*len(stable)/len(data):.1f}%)')
    
    if stable:
        print(f'\\n  Stable structures:')
        for s in stable[:10]:  # Show first 10
            print(f\"    • {s['formula']}: E_hull = {s['energy_above_hull']:.4f} eV/atom\")
        if len(stable) > 10:
            print(f'    ... and {len(stable)-10} more')
    
    if data:
        e_hulls = [d['energy_above_hull'] for d in data]
        import numpy as np
        print(f'\\n  E_hull statistics:')
        print(f'    Mean: {np.mean(e_hulls):.4f} eV/atom')
        print(f'    Median: {np.median(e_hulls):.4f} eV/atom')
        print(f'    Min: {np.min(e_hulls):.4f} eV/atom')
        print(f'    Max: {np.max(e_hulls):.4f} eV/atom')
"
fi

# MatterGen metrics (validity, novelty, etc.)
if [ -f "$OUTPUT_DIR/mattergen_metrics.json" ]; then
    echo ""
    echo "MatterGen evaluation metrics:"
    python -c "
import json
with open('$OUTPUT_DIR/mattergen_metrics.json', 'r') as f:
    data = json.load(f)
    print(f'  Total structures: {len(data)}')
    # Print available metrics
    if data and len(data) > 0:
        metrics = list(data[0].keys())
        print(f'  Available metrics: {metrics}')
"
fi

# Electride probability evaluation
echo ""
echo "=========================================="
echo "Evaluating Electride Probabilities"
echo "=========================================="

if [ -d "$CLASSIFIER_MODEL_DIR" ]; then
    python evaluate_with_classifier.py \
        --structures "$RESULTS_PATH" \
        --model_dir "$CLASSIFIER_MODEL_DIR" \
        --output "$OUTPUT_DIR/electride_probabilities.json"
    
    echo " Electride probabilities saved to: $OUTPUT_DIR/electride_probabilities.json"
else
    echo "  Classifier model not found at: $CLASSIFIER_MODEL_DIR"
    echo "  To train the classifier, run:"
    echo "    sbatch train_classifier.sh"
fi

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

