# MatterGen Electride Generation Project

This project uses MatterGen fine-tuning with a custom `is_electride` boolean property to generate and identify novel electride materials.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Electride Generation](#electride-generation)
   - [Binary Electride Generation](#binary-electride-generation)
   - [Ternary Electride Generation](#ternary-electride-generation)
5. [Parallel Generation](#parallel-generation)
6. [Evaluation](#evaluation)
7. [Results Analysis](#results-analysis)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Workflow

```
1. Dataset Preparation → Add is_electride property to mp_20 dataset
2. Fine-tuning        → Train MatterGen CSP model with is_electride property
3. Generation         → Generate binary and ternary electride structures using CSP mode
4. Evaluation         → Assess thermodynamic stability with MatterSim
5. Analysis           → Identify promising stable candidates for DFT validation
```

### Key Features

- Custom `is_electride` property integration with MatterGen
- CSP mode for targeted binary and ternary electride generation
- Automatic batch size optimization for efficient generation
- Parallel job submission for faster large-scale generation
- Resume capability for interrupted generation jobs
- MatterSim-based DFT-quality stability evaluation
- Comprehensive thermodynamic analysis pipeline

---

## Prerequisites

### Environment Setup

```bash
conda activate mattersim
```

### Required Files

- `electrides-reduced.csv` - Known electride materials from MP
- `mp_20/` directory - MatterGen mp_20 dataset
- MatterGen installation with modifications:
  - `is_electride` in `PROPERTY_SOURCE_IDS` (`mattergen/common/utils/globals.py`)
  - `is_electride.yaml` config (`mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/`)
  - `BooleanEmbeddingVector` class (`mattergen/property_embedding.py`)

---

## Dataset Preparation

### Step 1: Add is_electride Property to Dataset

The `add_field.py` script adds the `is_electride` boolean property to the mp_20 dataset.

```bash
cd ~/SOFT/mattergen_test/
python add_field.py
```

**Expected output:**
```
Extracted 145 material IDs to electride_material_ids.txt
Number of matched material_ids: X in mp_20/train.csv
Number of matched material_ids: Y in mp_20/val.csv
Number of matched material_ids: Z in mp_20/test.csv
Added is_electride field to mp_20/train.csv
Added is_electride field to mp_20/val.csv
Added is_electride field to mp_20/test.csv
```

**What it does:**
1. Extracts material IDs from `electrides-reduced.csv`
2. Adds `is_electride` column to all mp_20 CSV files (1 for electrides, 0 otherwise)
3. Saves matched IDs to `matched_*_ids.txt` files

### Step 2: Verify the Addition

```bash
# Check column exists
head -1 mp_20/train.csv | grep -o "is_electride"

# Count electrides per split
echo "Train electrides:" $(awk -F',' 'NR>1 && $NF==1' mp_20/train.csv | wc -l)
echo "Val electrides:" $(awk -F',' 'NR>1 && $NF==1' mp_20/val.csv | wc -l)
echo "Test electrides:" $(awk -F',' 'NR>1 && $NF==1' mp_20/test.csv | wc -l)
```

### Step 3: Convert CSV to Dataset Format

```bash
# On cluster
ssh HPC_HOST
conda activate mattersim
cd ~/SOFT/mattergen_test/

# Convert to MatterGen format
csv-to-dataset --csv-folder mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

### Step 4: Run Fine-tuning (Optional)

```bash
sbatch mattergen_finetune.sh
```

The fine-tuning script uses:
```bash
export PROPERTY1=is_electride
export PROPERTY2=metal_nonmetal

mattergen-finetune \
    adapter.pretrained_name=mattergen_base \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${PROPERTY1}=$PROPERTY1 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${PROPERTY2}=$PROPERTY2 \
    data_module.properties=[${PROPERTY1},${PROPERTY2}] \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    trainer.accelerator=gpu \
    trainer.precision=32 \
    trainer.accumulate_grad_batches=4
```

---

## Electride Generation

Generate binary and ternary electride crystal structures using MatterGen's CSP (Crystal Structure Prediction) mode with a fine-tuned model.

### Overview

The electride generation workflow systematically explores composition space to find potential electride materials:

1. **Search compositions** with excess valence electrons (potential electrides)
2. **Generate structures** for each composition using MatterGen CSP mode
3. **Evaluate stability** using MatterSim (DFT-quality potential)
4. **Analyze results** to identify promising candidates

---

## Binary Electride Generation

Binary electrides follow the simpler formula **A_l C_n** where:
- **A**: Electropositive metals (Group I, II, or III elements like Li, Na, K, Ca, Sr, Ba, Sc, Y)
- **C**: Electronegative non-metals (Group V, VI, or VII elements like N, P, O, S, F, Cl, Br, I)

### Step 1: Search for Binary Electride Compositions

```bash
cd ~/SOFT/mattergen_test/binary_electride

# Run composition search
python search_binary_electrides.py
```

**What it does:**
- Searches combinations of A-C binary compositions
  - **A**: Group I/II/III metals (Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba, Sc, Y, Be)
  - **C**: Group V/VI/VII non-metals (N, P, O, S, F, Cl, Br, I)
- Calculates excess valence electrons: `val_A * l - val_C * n`
- Keeps compositions with 0.1-4.0 excess electrons (electride range)
- Limits to max 20 atoms per composition

**Output files:**
```
binary_electride_compositions.json  # Full data with metadata
binary_electride_compositions.txt   # Just formulas for viewing
```

**Example compositions found:**
```
Li4N1   | 2.0 e⁻ | 5 atoms
Li5N1   | 2.0 e⁻ | 6 atoms
Li3O1   | 1.0 e⁻ | 4 atoms
K3N1    | 1.0 e⁻ | 4 atoms
Ca3N2   | 1.0 e⁻ | 5 atoms
```

### Step 2: Configure and Run Binary Generation

Edit `binary_electride/generate_binary_csp.sh` to set parameters:

```bash
CSP_MODEL="$HOME/SOFT/mattergen_test/outputs/singlerun/2025-10-16/18-55-08"
STRUCTURES_PER_ATOM=2.0   # Structures per atom
MAX_COMPOSITIONS=-1       # -1 = process all compositions
START_INDEX=0             # Starting index
```

Submit the job:

```bash
# Transfer to cluster
scp -r binary_electride/ HPC_HOST:~/SOFT/mattergen_test/

# On cluster
cd ~/SOFT/mattergen_test/binary_electride
mkdir -p logs
sbatch generate_binary_csp.sh

# Monitor progress
squeue -u $USER
tail -f logs/gen_bin_ele_csp_*.out
```

**Output structure:**
```
results/binary_csp_electrides/
├── Li4N1_structures/
│   ├── generated_crystals_cif.zip
│   └── generated_crystals.extxyz
├── Li5N1_structures/
│   ├── generated_crystals_cif.zip
│   └── generated_crystals.extxyz
├── generation_statistics.json
└── generation_summary.json
```

---

## Ternary Electride Generation

Ternary electrides follow the formula **A_l B_m C_n** where:
- **A**: Electropositive metals (Group I, II, or III)
- **B**: Semi-metals/metalloids (Group III or IV elements like Al, Si, Ge)
- **C**: Electronegative non-metals (Group V, VI, or VII)

### Step 1: Search for Ternary Electride Compositions

```bash
cd ~/SOFT/mattergen_test/ternary_electride

# Run composition search
python search_ternary_electrides.py
```

**What it does:**
- Searches combinations of A-B-C ternary compositions
  - **A**: Group I/II/III metals + lanthanides (Li, Na, K, Ca, Sr, Ba, La, etc.)
  - **B**: Group III/IV semi-metals (Al, Si, Ge, etc.)
  - **C**: Group V/VI/VII non-metals (N, O, F, P, S, etc.)
- Calculates excess valence electrons
- Keeps compositions with 0.1-4.0 excess electrons (electride range)
- Limits to max 20 atoms per composition

**Output files:**
```
ternary_electride_compositions.json  # Full data with metadata
ternary_electride_compositions.txt   # Just formulas for viewing
```

**Example compositions found:**
```
Li3AlN2  | 2.0 e⁻ | 6 atoms
Na3AlN2  | 2.0 e⁻ | 6 atoms  
Ca2SiO3  | 1.0 e⁻ | 6 atoms
K3GaN2   | 2.0 e⁻ | 6 atoms
```

### Step 2: Locate Fine-tuned CSP Model

```bash
# Find the fine-tuned model directory
find outputs/singlerun -type d -name "20*" | sort -r | head -1

# Example output:
# outputs/singlerun/2025-10-16/18-55-08
```

### Step 3: Configure Generation Script

Edit `ternary_electride/generate_ternary_csp.sh`:

```bash
# Update CSP model path
CSP_MODEL="$HOME/SOFT/mattergen_test/outputs/singlerun/2025-10-16/18-55-08"

# Set generation parameters
STRUCTURES_PER_ATOM=2.0   # Structures per atom
MAX_COMPOSITIONS=-1       # -1 = process all compositions
START_INDEX=0             # Starting index (for partial runs)
```

**Key Features:**
- **Automatic batch size**: Calculated per supercell to minimize waste
- **Even distribution**: Structures distributed equally across supercell sizes
- **Resume capability**: Automatically skips already-generated compositions
- **Supercell expansion**: Each composition expanded to multiple cell sizes (up to 20 atoms)

**Example:** For composition Li1B1N1 with 60 structures:
- Expands to 6 supercells: Li1B1N1 (3 atoms) to Li6B6N6 (18 atoms)
- Distributes: 10 structures per supercell
- Auto batch_size: 10 (divides evenly)
- Total: 60 structures

### Step 4: Run Generation

```bash
# Transfer to cluster
scp -r ternary_electride/ HPC_HOST:~/SOFT/mattergen_test/

# On cluster
cd ~/SOFT/mattergen_test/ternary_electride
mkdir -p logs

# Submit generation job
sbatch generate_ternary_csp.sh

# Monitor progress
squeue -u $USER
tail -f logs/gen_csp_*.out
```

**Generation Progress:**
The script will print detailed progress for each composition:
```
[1/200] Generating: Li3AlN2
  Configuration:
    Supercells: 6
    Target total structures: 60
    Distribution: 10 structures per supercell (even split)
    Auto batch_size: 10
    Actual total structures: 60
  Supercell sizes:
    1x: Li3AlN2 (6 atoms) - 10 structures
    2x: Li6Al2N4 (12 atoms) - 10 structures
    3x: Li9Al3N6 (18 atoms) - 10 structures
```

**Resume Capability:**
If job is interrupted, simply rerun the script. It will automatically skip compositions that already have generated structures.

**Output Structure:**
```
results/ternary_csp_electrides/
├── Li3AlN2_structures/
│   ├── generated_crystals_cif.zip
│   └── generated_crystals.extxyz
├── Na3AlN2_structures/
│   ├── generated_crystals_cif.zip
│   └── generated_crystals.extxyz
├── generation_statistics.json
├── generation_summary.json
└── failed_compositions.txt (if any failed)
```

### Step 5: Review Generation Results

```bash
# Check summary
cat results/ternary_csp_electrides/generation_summary.json

# Count total structures
find results/ternary_csp_electrides -name "*.cif" | wc -l
```

---

## Parallel Generation

For large composition sets, single GPU generation can be slow. Speed up generation by running multiple independent jobs in parallel.

### Why Parallel Generation?

**Performance comparison:**
- Single job: ~50 compositions/hour
- 5 parallel jobs: ~250 compositions/hour (5x speedup)
- 10 parallel jobs: ~500 compositions/hour (10x speedup)

**Note:** MatterGen doesn't support multi-GPU parallelization within a single job. Instead, run multiple single-GPU jobs processing different composition ranges simultaneously.

### Using Parallel Submission Scripts

Each workflow includes a pre-configured `submit_parallel_jobs.sh` script:

```bash
# For binary electrides
cd ~/SOFT/mattergen_test/binary_electride
./submit_parallel_jobs.sh

# For ternary electrides
cd ~/SOFT/mattergen_test/ternary_electride
./submit_parallel_jobs.sh
```

**What it does:**
1. Auto-detects total compositions from JSON file
2. Splits compositions into batches (default 200 per job)
3. Submits multiple SLURM jobs in parallel
4. Each job gets 1 GPU and processes its batch
5. All jobs write to same output directory (resume-safe)

**Customizing batch size:**

Edit `submit_parallel_jobs.sh`:
```bash
COMPOSITIONS_PER_JOB=200  # Smaller = more jobs, faster completion
```

Examples:
- 50 per job = max parallelization (many jobs)
- 100 per job = good balance (~10 jobs for 1000 compositions)
- 200 per job = conservative (fewer GPUs needed)

### Monitoring Parallel Jobs

```bash
# Check all your jobs
squeue -u $USER

# Count completed compositions
ls -d results/binary_csp_electrides/*_structures | wc -l

# Check progress of specific job
tail -f binary_electride/logs/gen_bin_ele_csp_batch0_*.out

# See job statistics
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed -S today
```

### Handling Failures

The resume capability makes parallel jobs safe:

```bash
# If a job fails or times out, resubmit with same parameters
sbatch generate_binary_csp.sh

# It will automatically:
# 1. Skip already-completed compositions
# 2. Continue from where it stopped
# 3. Avoid conflicts with other running jobs
```

### Resource Planning

**Binary Electrides (~400 compositions):**
- 4 parallel jobs × 100 compositions = ~2 hours total

**Ternary Electrides (~1000 compositions):**
- 5 parallel jobs × 200 compositions = ~4 hours total
- 10 parallel jobs × 100 compositions = ~2 hours total

### Cluster Etiquette

Before submitting many jobs:

```bash
# Check GPU availability
sinfo -p GPU -o "%P %a %l %D %T %N %G"

# Start with 3-5 jobs
# Monitor if they start quickly
# Submit more if GPUs available
```

**Be a good citizen:**
- Don't monopolize all GPUs
- Use appropriate time limits
- Cancel jobs you don't need: `scancel <job_id>`

---

## Evaluation

Evaluate generated structures for thermodynamic stability using MatterSim.

### Step 1: Configure Evaluation Script

Edit `mattergen_evaluate.sh` or use the provided script for ternary electride results:

```bash
# For ternary electride results
cd ~/SOFT/mattergen_test
# The script will process all compositions in results/ternary_csp_electrides/
```

**Evaluation configuration:**
- **GPU-accelerated:** Uses GPU nodes for fast MatterSim calculations
- **Automatic hull analysis:** Computes energy above hull with MP competing phases
- **Batch processing:** Evaluates all compositions systematically

### Step 2: Run Evaluation

```bash
# On cluster
cd ~/SOFT/mattergen_test
sbatch mattergen_evaluate.sh

# Monitor progress
tail -f logs/mattergen_eval_*.out
```

**What it does:**
1. Relaxes structures using MatterSim on GPU (DFT-quality machine learning potential)
2. Computes energy above hull using MP competing phases
   - Fetches competing phases from Materials Project
   - Relaxes competing phases with MatterSim for consistency
   - Calculates thermodynamic stability

**Performance:** GPU acceleration provides ~10-50x speedup for MatterSim relaxations compared to CPU.

**Output files:**
```
results/ternary_csp_electrides/*/
├── stability.json                 # Energy and stability metrics
├── relaxed_structures.extxyz      # Relaxed atomic structures
└── stability_analysis.json        # Detailed hull analysis
```

### Understanding Stability Metrics

**Energy per atom:**
- DFT-quality formation energy (eV/atom) from MatterSim
- Lower values indicate more stable structures
- Used to calculate energy above hull

**Energy above hull (E_hull):**
- Thermodynamic stability relative to competing phases
- Key metric for synthesizability assessment

| E_hull (eV/atom) | Interpretation |
|------------------|----------------|
| 0.000 | Stable phase - On the convex hull |
| 0.000 - 0.025 | Highly likely synthesizable |
| 0.025 - 0.050 | Likely synthesizable with appropriate conditions |
| 0.050 - 0.100 | Potentially synthesizable (metastable) |
| > 0.100 | Unlikely to be thermodynamically stable |

**Example output:**
```json
{
  "cif_file": "supercell_1x_6atoms/structure_005.cif",
  "formula": "Li3AlN2",
  "num_sites": 6,
  "energy_per_atom": -3.456,
  "e_above_hull": 0.023,
  "hull_competing_phases": ["Li3N", "AlN", "Li"],
  "is_stable": false,
  "is_on_hull": false
}
```

---

## Results Analysis

### Quick Summary

```bash
# View evaluation summary
cat logs/mattergen_eval_*.out | grep -A 20 "EVALUATION SUMMARY"

# View generation summary
cat results/ternary_csp_electrides/generation_summary.json
```

### Extract Stable Candidates

Use the provided script to filter structures by stability:

```bash
cd ~/SOFT/mattergen_test

python extract_stable_candidates.py \
    --results-dir results/ternary_csp_electrides \
    --max-e-hull 0.100 \
    --output stable_candidates.json
```

**Options:**
- `--max-e-hull`: Maximum energy above hull (default: 0.100 eV/atom)
- `--min-structures`: Minimum structures per composition to report
- `--output`: Output file for filtered results

### Analyze Stable Structures

```python
import json
from collections import defaultdict

# Load stability results
with open('stable_candidates.json') as f:
    stable = json.load(f)

# Group by composition
by_composition = defaultdict(list)
for s in stable:
    by_composition[s['formula']].append(s)

# Print top compositions by number of stable structures
top_comps = sorted(by_composition.items(), 
                   key=lambda x: len(x[1]), reverse=True)[:10]

print("Top 10 Compositions by Stable Structure Count:")
for formula, structures in top_comps:
    min_hull = min(s['e_above_hull'] for s in structures)
    print(f"{formula}: {len(structures)} stable structures, "
          f"best E_hull = {min_hull:.4f} eV/atom")
```

### Find Most Stable Structures

```python
# Sort all structures by E_hull
sorted_structures = sorted(stable, key=lambda x: x['e_above_hull'])

print("Top 20 Most Stable Structures:")
for i, s in enumerate(sorted_structures[:20], 1):
    print(f"{i:2d}. {s['formula']:15s} | "
          f"E_hull: {s['e_above_hull']:7.4f} eV/atom | "
          f"E/atom: {s['energy_per_atom']:8.4f} eV")
```

### Expected Results

From ~200 ternary compositions with 60 structures each (~12,000 total):
- ~5-15% thermodynamically stable (E_hull < 0.1 eV/atom)
- ~600-1,800 potentially synthesizable structures
- ~10-50 compositions with multiple stable polymorphs
- ~1-5 compositions on or near the convex hull

### Next Steps

1. **Select promising candidates:**
   - Focus on E_hull < 0.025 eV/atom (highly synthesizable)
   - Prioritize compositions with excess electrons
   - Look for multiple stable polymorphs

2. **Structural analysis:**
   - Visualize structures (VESTA, Ovito, etc.)
   - Identify common structural motifs
   - Compare with known electrides

3. **DFT validation:**
   - Perform high-accuracy DFT on top 20-30 candidates
   - Calculate electronic structure (band structure, DOS)
   - Analyze electron localization (ELF, COHP)
   - Confirm electride character (localized electrons in interstitial regions)

4. **Experimental candidates:**
   - Rank by synthesizability (E_hull and chemical feasibility)
   - Check for similar known materials in literature
   - Assess synthetic accessibility

---

## Troubleshooting

### Dataset Preparation

**No matches found:**
- Check material_id column names match exactly
- Verify electride material IDs format (e.g., "mp-XXXXX")
- Inspect `electride_material_ids.txt` content

**csv-to-dataset fails:**
- Verify correct conda environment
- Check `is_electride` property registered in `globals.py`
- Confirm YAML config in correct location

### Composition Search

**No compositions found:**
- Check element groups defined in `search_ternary_electrides.py`
- Adjust `excess_electron_range` parameter
- Verify max_atoms setting

**Too many compositions:**
```python
# In search_ternary_electrides.py, reduce search space:
compositions = search_ternary_electrides(
    max_atoms=15,  # Reduce from 20
    max_compositions=500  # Cap at 500
)
```

### Generation

**Out of GPU memory:**
```bash
# Reduce structures per composition in generate_ternary_csp.sh
STRUCTURES_PER_ATOM=1.0  # Reduce from 2.0
# Batch size is auto-calculated and will adjust accordingly
```

**CSP model not found:**
```bash
# Verify model path in generate_ternary_csp.sh
find outputs/singlerun -type d -name "20*" | sort -r | head -1
```

**Generation interrupted:**
```bash
# Simply rerun the script - it will automatically skip completed compositions
sbatch ternary_electride/generate_ternary_csp.sh
```

**Some compositions fail:**
- Check `failed_compositions.txt` in results directory
- Review error messages for specific failures
- May need to retry with different parameters or skip problematic compositions

### Evaluation

**Evaluation too slow:**
```bash
# Use --relax=False for quick checks
# Or reduce number of structures
```

**MatterSim errors:**
```bash
# Verify mattersim installation
conda activate mattersim
python -c "import mattersim; print(mattersim.__version__)"
```

**Missing terminal entries error:**
This occurs when MP doesn't have data for a chemical system. The script now handles this gracefully by:
- Skipping hull analysis for systems without MP data
- Recording structures with `energy_above_hull: None`
- Continuing with other chemical systems

**MP API structure fetch errors:**
The script now robustly handles different MP API response formats:
- Checks if structure is attached to entry first
- Handles both single structures and lists
- Separates fetch errors from relaxation errors
- Shows detailed error breakdown

---

## Advanced Topics

### Manual Parallel Job Submission

For more control over job ranges, manually edit and submit jobs:

```bash
# Edit generate_ternary_csp.sh for each range
START_INDEX=0
MAX_COMPOSITIONS=100

# Submit multiple jobs with different ranges
sbatch generate_ternary_csp.sh  # Edit START_INDEX between submissions
```

Alternatively, use the automated `submit_parallel_jobs.sh` script (see [Parallel Generation](#parallel-generation) section).

### Custom Composition Search

Modify `search_ternary_electrides.py` to explore specific chemical spaces:

```python
# Example 1: Focus on alkali metal nitrides
GROUP_A = ['Li', 'Na', 'K', 'Rb', 'Cs']
GROUP_B = ['Al', 'Si']
GROUP_C = ['N']

# Example 2: Explore rare earth oxides
GROUP_A = ['La', 'Ce', 'Pr', 'Nd', 'Y']
GROUP_B = ['Si', 'Ge', 'Sn']
GROUP_C = ['O']
```

### Batch Re-evaluation

Re-evaluate specific compositions with different settings:

```bash
# Evaluate single composition with custom parameters
python evaluate_stability_hull.py \
    --structures-path results/ternary_csp_electrides/Li3AlN2_structures \
    --output-file Li3AlN2_stability_custom.json \
    --use-mattersim \
    --relax-structures
```

---

## Workflow Comparison

### Generation Strategies

| Strategy | Structures | Diversity | Computation | Best For |
|----------|------------|-----------|-------------|----------|
| Property Conditioning | Large batches | High | Moderate | Exploration, discovery |
| CSP Mode | Per composition | Targeted | High | Specific compositions, systematic search |
| Hybrid | Mixed | Balanced | Variable | Iterative refinement |

**This workflow uses CSP mode** for systematic exploration of ternary electride composition space.

---

## File Structure

```
mattergen_test/
├── README.md                          # This file
├── add_field.py                       # Add is_electride to dataset
├── evaluate_stability_hull.py         # Stability evaluation script
├── extract_stable_candidates.py       # Filter stable structures
├── mattergen_finetune.sh              # Fine-tuning job script
├── mattergen_evaluate.sh              # Evaluation job script
├── electrides-reduced.csv             # Known electrides from MP
├── mp_20/                             # Dataset with is_electride property
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── outputs/                           # Fine-tuning outputs
│   └── singlerun/
│       └── 2025-10-16/18-55-08/      # Fine-tuned CSP model
├── binary_electride/                  # Binary electride workflow
│   ├── search_binary_electrides.py    # Composition search (A_l C_n)
│   ├── generate_structures_batch.py   # Batch structure generation
│   ├── generate_binary_csp.sh         # Generation job script
│   ├── submit_parallel_jobs.sh        # Parallel job submission
│   ├── summarize_generation.py        # Generate summaries
│   ├── binary_electride_compositions.json  # Target compositions
│   └── logs/                          # Generation logs
├── ternary_electride/                 # Ternary electride workflow
│   ├── search_ternary_electrides.py   # Composition search (A_l B_m C_n)
│   ├── generate_structures_batch.py   # Batch structure generation
│   ├── generate_ternary_csp.sh        # Generation job script
│   ├── submit_parallel_jobs.sh        # Parallel job submission
│   ├── summarize_generation.py        # Generate summaries
│   ├── ternary_electride_compositions.json  # Target compositions
│   └── logs/                          # Generation logs
├── results/                           # Generation results
│   ├── binary_csp_electrides/         # Binary CSP results
│   │   ├── Li4N1_structures/
│   │   │   ├── generated_crystals_cif.zip
│   │   │   └── generated_crystals.extxyz
│   │   ├── generation_statistics.json
│   │   └── generation_summary.json
│   └── ternary_csp_electrides/        # Ternary CSP results
│       ├── Li3AlN2_structures/
│       │   ├── generated_crystals_cif.zip
│       │   └── generated_crystals.extxyz
│       ├── generation_statistics.json
│       ├── generation_summary.json
│       └── failed_compositions.txt
└── logs/                              # Job logs
    ├── mattergen_finetune_*.out
    ├── gen_csp_*.out
    └── mattergen_eval_*.out
```

---

## Reference

For more details on MatterGen:
- MatterGen documentation: https://github.com/microsoft/mattergen
- MatterSim documentation: https://github.com/microsoft/mattersim

For questions or issues specific to this workflow:
- Check relevant log files in `logs/`
- Verify environment and dependencies
- Review error messages in `.err` files

