# Binary Electride Generation with MatterGen CSP Mode

This directory contains scripts for generating binary electride crystal structures using MatterGen's Crystal Structure Prediction (CSP) mode.

## Overview

Binary electrides follow the formula **A_l C_n** where:
- **A**: Electropositive metals (Group I, II, or III elements)
- **C**: Electronegative non-metals (Group V, VI, or VII elements)

The workflow searches for compositions with excess electrons that could form electride structures, then generates candidate crystal structures using MatterGen.

## Workflow Steps

### 1. Search for Binary Electride Compositions

```bash
python search_binary_electrides.py
```

This creates:
- `binary_electride_compositions.json` - Full composition data with excess electrons
- `binary_electride_compositions.txt` - Simple list of formulas

**Key parameters** (in `main()` function):
- `max_atoms=15` - Maximum atoms per unit cell
- `excess_electron_range=(0.1, 2.0)` - Excess electrons for electride formation
- `max_compositions=-1` - Generate all compositions (-1 means no limit)

### 2. Generate Crystal Structures

#### Option A: Single Job (Small Scale)

```bash
sbatch generate_binary_csp.sh
```

**Key parameters** in the script:
- `STRUCTURES_PER_COMP=60` - Number of structures per composition
- `MAX_COMPOSITIONS=-1` - Process all compositions (-1 means all)
- `START_INDEX=0` - Starting composition index

**Note on structure counts**: Each composition is expanded into multiple supercells (up to max 20 atoms). The script automatically distributes structures evenly across all supercell sizes and calculates optimal batch sizes.

#### Option B: Parallel Jobs (Large Scale)

For faster processing of many compositions:

```bash
./submit_parallel_jobs.sh
```

This submits multiple independent jobs that process different composition ranges in parallel. See `PARALLEL_GENERATION_GUIDE.md` for details.

**Resume capability**: The script automatically skips compositions that already have generated structures, allowing you to resubmit if a job times out.

### 3. Review Results

Generated structures are saved to `../results/binary_csp_electrides/`:

```
../results/binary_csp_electrides/
├── Li1N1_structures/
│   ├── generated_crystals_cif.zip
│   └── generated_crystals.extxyz
├── Li2N1_structures/
│   ├── generated_crystals_cif.zip
│   └── generated_crystals.extxyz
├── generation_statistics.json
└── generation_summary.json
```

## File Descriptions

### Search Scripts
- `search_binary_electrides.py` - Composition search based on valence electron counting

### Generation Scripts
- `generate_structures_batch.py` - Main generation script with supercell expansion
- `generate_binary_csp.sh` - SLURM job script for structure generation
- `submit_parallel_jobs.sh` - Helper to submit multiple parallel jobs
- `summarize_generation.py` - Creates summary statistics after generation

### Output Files
- `binary_electride_compositions.json` - Compositions with metadata
- `binary_electride_compositions.txt` - Simple formula list
- `generation_statistics.json` - Per-composition generation stats
- `generation_summary.json` - Overall generation summary
- `failed_compositions.txt` - List of failed compositions (if any)

## CSP Model Configuration

The generation script uses a fine-tuned CSP model. Update the path in `generate_binary_csp.sh`:

```bash
CSP_MODEL="$HOME/SOFT/mattergen_test/outputs/singlerun/2025-10-16/18-55-08"
```

## Supercell Expansion

For each composition, the script automatically generates multiple supercells up to 20 atoms. For example, `Li1N1` generates:
- Li1N1 (2 atoms)
- Li2N2 (4 atoms)
- Li3N3 (6 atoms)
- ...
- Li10N10 (20 atoms)

The `STRUCTURES_PER_COMP` parameter is distributed evenly across all supercell sizes.

## Troubleshooting

### GPU Memory Issues
If you encounter OOM errors, reduce `STRUCTURES_PER_COMP` in the generation script.

### Job Timeout
Use `submit_parallel_jobs.sh` to split the work across multiple jobs, each processing fewer compositions.

### Resume Generation
Simply resubmit the job - it will automatically skip compositions that already have generated structures.

## Next Steps

After generation:
1. Review `generation_summary.json` for statistics
2. Evaluate structures for stability using MatterSim
3. Filter candidates by energy above hull and electride probability

