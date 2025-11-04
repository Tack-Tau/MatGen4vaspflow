# Generating Ternary Electride Structures with MatterGen

## Overview

This workflow generates crystal structures for potential ternary electride compositions using MatterGen's **Crystal Structure Prediction (CSP) mode** with the base model (no fine-tuning required).

**Strategy:**
1. Search for compositions with excess valence electrons (potential electrides)
2. Generate ~20 structures per composition using MatterGen CSP mode
3. Evaluate structures for stability and electride character

---

## Prerequisites

 ASE and Pymatgen installed (already in your `fplibenv` locally, `mattersim` on cluster)
 MatterGen base model accessible
 GPU access on cluster

---

## Step 1: Search for Ternary Electride Compositions

### On Local Machine (or Cluster):

```bash
cd ~/SOFT/mattergen_test

# Run composition search
python search_ternary_electrides.py
```

**What it does:**
- Searches through combinations of: `A_l B_m C_n`
  - **A**: Group I/II/III metals + lanthanides (Li, Na, K, Ca, Sr, Ba, La, etc.)
  - **B**: Group III/IV semi-metals (Al, Si, Ge, etc.)
  - **C**: Group V/VI/VII non-metals (N, O, F, P, S, etc.)
- Calculates excess valence electrons: `excess = val(A)×l + val(B)×m - val(C)×n`
- Keeps compositions with 0.1-4.0 excess electrons (electride range)
- Limits to max 20 atoms per composition
- Finds up to 1000 valid compositions

**Output files:**
```
ternary_electride_compositions.json  # Full data with metadata
ternary_electride_compositions.txt   # Just formulas for easy viewing
```

**Example compositions found:**
```
Li3AlN2     | 2.0 e⁻ | 6 atoms
Na3AlN2     | 2.0 e⁻ | 6 atoms  
Ca2SiO3     | 1.0 e⁻ | 6 atoms
K3GaN2      | 2.0 e⁻ | 6 atoms
```

### Customize the Search:

Edit `search_ternary_electrides.py` to adjust:

```python
compositions = search_ternary_electrides(
    max_atoms=20,                      # Change max atoms per unit cell
    excess_electron_range=(0.1, 4.0),  # Change e⁻ range
    max_compositions=1000              # Change number of compositions
)
```

---

## Step 2: Transfer to Cluster

```bash
# Transfer composition file and generation script
scp ternary_electride_compositions.json HPC_HOST:~/SOFT/mattergen_test/
scp generate_ternary_electrides.sh HPC_HOST:~/SOFT/mattergen_test/
```

---

## Step 3: Generate Structures with MatterGen CSP Mode

### On Cluster:

```bash
ssh HPC_HOST
cd ~/SOFT/mattergen_test

# Make script executable
chmod +x generate_ternary_electrides.sh

# Submit generation job
sbatch generate_ternary_electrides.sh

# Monitor progress
tail -f logs/mattergen_ternary_<jobid>.out
```

### What Happens:

For each composition (e.g., `Li3AlN2`):
1. MatterGen runs in **CSP mode** (conditioned on exact composition)
2. Generates 20 structures (adjustable via `STRUCTURES_PER_COMP`)
3. Saves structures as CIF files in `results/ternary_electrides/Li3AlN2_structures/`

**Key Parameters in Script:**

```bash
MODEL="mattergen_base"           # Base model (no fine-tuning)
STRUCTURES_PER_COMP=60           # Target structures per composition
MAX_COMPOSITIONS=-1              # Process all compositions (-1 = all)
```

**Note on Structure Counts:**
- Each composition is expanded to multiple supercells (e.g., Li1B1N1, Li2B2N2, ..., Li6B6N6)
- Structures are distributed as evenly as possible across all supercells
- Batch size is automatically calculated per supercell to minimize waste
- Handles remainders intelligently:
  - If not evenly divisible, some supercells get 1 extra structure
  - Difference between any two supercells is at most 1 structure
  
**Examples:**
- 60 structures, 6 supercells: Each gets 10 structures (even split)
- 65 structures, 6 supercells: 5 get 11 structures, 1 gets 10 structures
- 100 structures, 7 supercells: 2 get 15 structures, 5 get 14 structures

The algorithm prefers batch sizes that divide evenly (1, 2, 4, 5, 8, 10, 16, 20, 32, 64)

### CSP Mode vs Property Conditioning:

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| **CSP** | Chemical formula (e.g., Li3AlN2) | Structures with exact composition | Specific composition search |
| **Property** | Property value (e.g., is_electride=1) | Structures with that property | Exploration/discovery |

**We use CSP mode here** because we have specific target compositions from our search.

---

## Step 4: Review Results

```bash
# Check summary
cat results/ternary_electrides/generation_summary.json

# List generated compositions
ls results/ternary_electrides/*/

# Count structures
find results/ternary_electrides -name "*.cif" | wc -l
```

**Expected output structure:**
```
results/ternary_electrides/
├── Li3AlN2_structures/
│   ├── structure_000.cif
│   ├── structure_001.cif
│   ├── ...
│   └── structure_019.cif (20 total)
├── Na3AlN2_structures/
│   ├── structure_000.cif
│   ├── ...
├── generation_summary.json
└── failed_compositions.txt (if any failed)
```

---

## Step 5: Evaluate Generated Structures

### Option A: Evaluate All Compositions Together

```bash
# Combine all CIFs into one zip file
cd results/ternary_electrides
find . -name "*.cif" -print | zip all_ternary_structures.zip -@

# Evaluate with MatterSim
mattergen-evaluate \
    --structures_path=all_ternary_structures.zip \
    --relax=True \
    --potential_load_path="MatterSim-v1.0.0-5M.pth" \
    --save_as="all_stability.json"
```

### Option B: Evaluate Per Composition

Create a batch evaluation script:

```bash
for comp_dir in results/ternary_electrides/*_structures/; do
    comp_name=$(basename $comp_dir _structures)
    echo "Evaluating $comp_name..."
    
    # Zip structures for this composition
    cd $comp_dir
    zip ${comp_name}_structures.zip *.cif
    
    # Evaluate
    mattergen-evaluate \
        --structures_path=${comp_name}_structures.zip \
        --relax=True \
        --potential_load_path="MatterSim-v1.0.0-5M.pth" \
        --save_as="stability_${comp_name}.json"
    
    cd ../..
done
```

---

## Step 6: Filter and Analyze

### Find Most Stable Structures:

```python
import json
import glob
from collections import defaultdict

# Load all stability results
stability_files = glob.glob("results/ternary_electrides/*/stability_*.json")

best_structures = []

for stab_file in stability_files:
    with open(stab_file) as f:
        data = json.load(f)
    
    for structure in data:
        if structure.get('energy_per_atom') and structure['energy_per_atom'] != float('inf'):
            best_structures.append({
                'composition': stab_file.split('/')[-1].replace('stability_', '').replace('.json', ''),
                'energy': structure['energy_per_atom'],
                'e_hull': structure.get('e_above_hull', 999),
                'structure_id': structure.get('structure_id')
            })

# Sort by energy above hull
best_structures.sort(key=lambda x: x['e_hull'])

# Print top 20 most stable
print("TOP 20 MOST STABLE STRUCTURES:")
print("="*70)
for i, s in enumerate(best_structures[:20], 1):
    print(f"{i:2d}. {s['composition']:15s} | E_hull: {s['e_hull']:7.4f} eV/atom | E: {s['energy']:7.4f}")

# Filter synthesizable (E_hull < 0.1 eV/atom)
synthesizable = [s for s in best_structures if s['e_hull'] < 0.1]
print(f"\nPotentially synthesizable structures (E_hull < 0.1): {len(synthesizable)}")
```

### Predict Electride Likelihood:

```python
# Use trained classifier
import joblib
from train_electride_classifier import predict_on_generated_structures

clf = joblib.load("electride_classifier.joblib")

# Evaluate each composition directory
for comp_dir in glob.glob("results/ternary_electrides/*_structures/"):
    comp_name = comp_dir.split('/')[-2].replace('_structures', '')
    
    results = predict_on_generated_structures(
        clf,
        comp_dir,
        f"{comp_dir}/electride_probabilities.json"
    )
    
    # Print high-probability electrides
    high_prob = [r for r in results if r['electride_probability'] > 0.7]
    if high_prob:
        print(f"\n{comp_name}: {len(high_prob)}/{len(results)} likely electrides")
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SEARCH COMPOSITIONS (Local/Cluster)                      │
│    python search_ternary_electrides.py                      │
│    → ternary_electride_compositions.json (1000 compositions)│
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 2. GENERATE STRUCTURES (Cluster - GPU)                      │
│    sbatch generate_ternary_electrides.sh                    │
│    → 20 structures × 100 compositions = 2000 structures     │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 3. EVALUATE (Cluster - GPU)                                 │
│    • Stability (MatterSim)                                  │
│    • Electride probability (ML classifier)                  │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 4. FILTER & ANALYZE                                         │
│    • E_hull < 0.1 eV/atom (stable)                          │
│    • electride_prob > 0.7 (likely electride)                │
│    → Top candidates for experimental validation             │
└─────────────────────────────────────────────────────────────┘
```

---

## Advanced: Customize Element Selection

Edit `search_ternary_electrides.py` to try different element combinations:

```python
# Example 1: Focus on alkali metals only
GROUP_A = ['Li', 'Na', 'K', 'Rb', 'Cs']
GROUP_B = ['Al', 'Si']
GROUP_C = ['N', 'O']

# Example 2: Include rare earths more prominently
GROUP_A = ['La', 'Ce', 'Pr', 'Nd', 'Y']
GROUP_B = ['Si', 'Ge', 'Sn']
GROUP_C = ['N', 'O']

# Example 3: Search binary compositions (set GROUP_C to single element)
GROUP_A = ['Ca', 'Sr', 'Ba']
GROUP_B = ['Al']
GROUP_C = ['N']  # Only nitrides
```

---

## Troubleshooting

### Problem: Too many compositions (>10,000)

**Solution:** Reduce search space:
```python
compositions = search_ternary_electrides(
    max_atoms=15,  # Reduce from 20
    max_compositions=500  # Cap at 500
)
```

### Problem: Generation is too slow

**Solution:** Parallelize by composition:
```bash
# Split compositions into chunks and run multiple jobs
# Job 1: compositions 0-99
# Job 2: compositions 100-199
# etc.
```

Edit script parameter:
```bash
# In generate_ternary_electrides.sh, add offset:
START_IDX=0
MAX_COMPOSITIONS=100
# Then modify Python script to use compositions[START_IDX:START_IDX+MAX_COMPOSITIONS]
```

### Problem: Out of GPU memory

**Solution:** The batch size is auto-calculated. If you get OOM errors, reduce structures per composition:
```bash
STRUCTURES_PER_COMP=30  # Reduce from 60
# This will automatically use smaller batch sizes per supercell
```

### Problem: Some compositions fail to generate

**Solution:** Check `failed_compositions.txt` and retry:
```bash
# Failed compositions are saved automatically
cat results/ternary_electrides/failed_compositions.txt

# Manually retry specific compositions
mattergen-generate results/retry_Li3AlN2 \
    --model_path=mattergen_base \
    --sampling_config_name=csp \
    --target_compositions='[{"Li": 3, "Al": 1, "N": 2}]' \
    --batch_size=20 \
    --num_batches=1
```

---

## Expected Results

**From ~1000 compositions:**
- Generate ~20,000 structures (20 per composition)
- ~10-30% will be thermodynamically stable (E_hull < 0.1)
- ~2,000-6,000 stable structures
- Of these, filter by electride probability > 0.7
- **Final candidates:** 100-500 promising new electride structures

**Timeline:**
- Composition search: ~1 minute
- Structure generation: 6-24 hours (depends on # of compositions and GPU)
- Evaluation: 2-6 hours

---

## Citation

If you use this workflow, cite:
- **MatterGen:** Zeni et al., Nature (2025) - https://doi.org/10.1038/s41586-025-08628-5
- **MatterSim:** (cite if using for evaluation)

---

## Summary

This workflow enables **high-throughput computational discovery** of ternary electride materials by:
1.  Systematic composition space search
2.  AI-driven structure generation (MatterGen CSP mode)
3.  DFT-quality stability prediction (MatterSim)
4.  ML-based property prediction (electride classifier)

Good luck discovering new electrides! 🔬⚡✨

