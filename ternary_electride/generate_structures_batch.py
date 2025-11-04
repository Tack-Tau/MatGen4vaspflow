#!/usr/bin/env python3
"""
Generate crystal structures for multiple compositions using MatterGen CSP mode.

This script supports both:
- Fine-tuned CSP checkpoints: Pass checkpoint directory path
- Pretrained models: Pass model name (e.g., "mattergen_base")

This script calls mattergen-generate for each composition via subprocess.
"""

import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Dict
import argparse


def calculate_optimal_batch_size(structures_per_supercell: int) -> int:
    """
    Calculate optimal batch size to minimize rounding waste.
    
    Args:
        structures_per_supercell: Number of structures to generate per supercell
    
    Returns:
        Optimal batch size (power of 2 or common divisor)
    """
    # If very small, use it directly
    if structures_per_supercell <= 8:
        return structures_per_supercell
    
    # Preferred batch sizes (powers of 2 and common values)
    preferred_sizes = [64, 32, 20, 16, 10, 8, 5, 4, 2, 1]
    
    # Find largest batch size that divides evenly or is close
    best_batch_size = 1
    min_waste = float('inf')
    
    for batch_size in preferred_sizes:
        if batch_size > structures_per_supercell:
            continue
        
        # Calculate waste from rounding up
        num_batches = math.ceil(structures_per_supercell / batch_size)
        actual = num_batches * batch_size
        waste = actual - structures_per_supercell
        
        # Prefer exact divisors (no waste)
        if waste == 0:
            return batch_size
        
        # Otherwise minimize waste
        if waste < min_waste:
            min_waste = waste
            best_batch_size = batch_size
    
    return best_batch_size


def expand_composition_to_supercells(composition: Dict[str, int], max_atoms: int = 20) -> List[Dict[str, int]]:
    """
    Expand a reduced composition to all valid supercells up to max_atoms.
    
    For example, Li1B3P2 (6 atoms total) expands to:
    - 1x: Li1B3P2 (6 atoms)
    - 2x: Li2B6P4 (12 atoms)
    - 3x: Li3B9P6 (18 atoms)
    - 4x would be 24 atoms, exceeding max_atoms=20
    
    This allows MatterGen to explore different cell sizes in CSP mode.
    
    Args:
        composition: Reduced composition dict, e.g. {"Li": 1, "B": 3, "P": 2}
        max_atoms: Maximum total atoms allowed (mp_20 was trained on ≤20 atoms)
    
    Returns:
        List of composition dicts with different multipliers
    """
    # Calculate total atoms in reduced formula
    total_atoms_base = sum(composition.values())
    
    # Calculate how many multiples we can fit
    max_multiplier = max_atoms // total_atoms_base
    
    # Generate all valid supercells
    supercells = []
    for multiplier in range(1, max_multiplier + 1):
        supercell = {elem: count * multiplier for elem, count in composition.items()}
        total = sum(supercell.values())
        supercells.append(supercell)
    
    return supercells


def check_if_already_generated(output_dir: Path, min_structures: int = 1) -> bool:
    """
    Check if structures have already been generated for this composition.
    
    Args:
        output_dir: Output directory for composition
        min_structures: Minimum number of structures to consider complete
        
    Returns:
        True if already generated, False otherwise
    """
    if not output_dir.exists():
        return False
    
    # Check for combined output files
    extxyz_file = output_dir / "generated_crystals.extxyz"
    cif_zip = output_dir / "generated_crystals_cif.zip"
    
    # Consider complete if either file exists
    if extxyz_file.exists() or cif_zip.exists():
        return True
    
    return False


def generate_structures_for_composition(
    composition: Dict[str, int],
    formula: str,
    output_dir: Path,
    model_path: str,
    n_structures: int,
    max_atoms: int = 20,
    timeout: int = 1800,
    skip_if_exists: bool = True
) -> bool:
    """
    Generate structures for a single composition using MatterGen CSP mode.
    
    This function expands the composition to all valid supercells (up to max_atoms)
    and generates equal numbers of structures for EACH supercell individually.
    Batch size is automatically calculated to minimize rounding waste.
    
    Args:
        composition: Dict like {"Li": 3, "Al": 1, "N": 2}
        formula: String formula like "Li3AlN2"
        output_dir: Output directory for structures
        model_path: Path to MatterGen model checkpoint or pretrained name
                   - Checkpoint: "outputs/singlerun/2025-10-10/15-30-45"
                   - Pretrained: "mattergen_base" or "mp_20_base"
        n_structures: Total number of structures to generate
        max_atoms: Maximum atoms per cell (default: 20 for mp_20)
        timeout: Timeout in seconds (default 30 min)
        skip_if_exists: Skip generation if structures already exist
        
    Returns:
        True if successful, False otherwise
    """
    # Check if already generated (for resume capability)
    if skip_if_exists and check_if_already_generated(output_dir, min_structures=1):
        print(f"    Already generated (skipping)")
        return True
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expand composition to all valid supercells
    supercells = expand_composition_to_supercells(composition, max_atoms)
    
    # Distribute structures evenly across supercells, handling remainders
    base_structures = n_structures // len(supercells)
    remainder = n_structures % len(supercells)
    
    # First 'remainder' supercells get (base_structures + 1), rest get base_structures
    structures_per_supercell_list = []
    for i in range(len(supercells)):
        if i < remainder:
            structures_per_supercell_list.append(base_structures + 1)
        else:
            structures_per_supercell_list.append(base_structures)
    
    # Calculate batch sizes for the two groups
    if remainder > 0:
        batch_size_high = calculate_optimal_batch_size(base_structures + 1)
        batch_size_low = calculate_optimal_batch_size(base_structures)
    else:
        batch_size_high = calculate_optimal_batch_size(base_structures)
        batch_size_low = batch_size_high
    
    # Calculate actual total structures
    total_structures_to_generate = 0
    for n_struct in structures_per_supercell_list:
        batch_sz = batch_size_high if n_struct == base_structures + 1 else batch_size_low
        num_batches = max(1, math.ceil(n_struct / batch_sz))
        total_structures_to_generate += num_batches * batch_sz
    
    print(f"  Configuration:")
    print(f"    Supercells: {len(supercells)}")
    print(f"    Target total structures: {n_structures}")
    if remainder > 0:
        print(f"    Distribution: {remainder} supercells with {base_structures + 1} structures, {len(supercells) - remainder} with {base_structures}")
        print(f"    Auto batch_size: {batch_size_high} (high) / {batch_size_low} (low)")
    else:
        print(f"    Distribution: {base_structures} structures per supercell (even split)")
        print(f"    Auto batch_size: {batch_size_high}")
    print(f"    Actual total structures: {total_structures_to_generate}")
    
    if total_structures_to_generate != n_structures:
        print(f"    Note: {total_structures_to_generate - n_structures} extra structures due to batch rounding")
    
    # Print supercell info
    print(f"  Supercell sizes:")
    for i, sc in enumerate(supercells, 1):
        total = sum(sc.values())
        sc_formula = ''.join(f"{elem}{count}" for elem, count in sorted(sc.items()))
        n_struct = structures_per_supercell_list[i-1]
        print(f"    {i}x: {sc_formula} ({total} atoms) - {n_struct} structures")
    
    # Determine if model_path is a checkpoint directory or pretrained model name
    is_checkpoint = ('/' in model_path or Path(model_path).exists())
    
    # Generate structures for each supercell to temporary directories
    temp_dir = output_dir / ".temp_generation"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    total_generated = 0
    all_cif_files = []
    all_extxyz_entries = []
    
    for supercell_idx, supercell in enumerate(supercells, 1):
        sc_formula = ''.join(f"{elem}{count}" for elem, count in sorted(supercell.items()))
        sc_atoms = sum(supercell.values())
        
        # Get target structures for this specific supercell
        target_structures = structures_per_supercell_list[supercell_idx - 1]
        
        # Create temporary subdirectory for this supercell
        supercell_dir = temp_dir / f"supercell_{supercell_idx}x_{sc_atoms}atoms"
        supercell_dir.mkdir(parents=True, exist_ok=True)
        
        # Format composition as JSON list with single supercell
        comp_arg = json.dumps([supercell])
        
        # Calculate batch size and number of batches for this supercell
        current_batch_size = batch_size_high if target_structures == base_structures + 1 else batch_size_low
        num_batches = max(1, math.ceil(target_structures / current_batch_size))
        actual_structures = num_batches * current_batch_size
        
        # Build mattergen-generate command
        cmd = [
            "mattergen-generate",
            str(supercell_dir),
        ]
        
        if is_checkpoint:
            cmd.append(f"--model_path={model_path}")
        else:
            cmd.append(f"--pretrained_name={model_path}")
        
        cmd.extend([
            "--sampling_config_name=csp",
            f"--target_compositions={comp_arg}",
            f"--batch_size={current_batch_size}",
            f"--num_batches={num_batches}",
            "--record_trajectories=False"
        ])
        
        if not is_checkpoint:
            cmd.extend([
                "--trainer.accelerator=gpu",
                "--trainer.devices=1",
                "--trainer.precision=32"
            ])
        
        print(f"    [{supercell_idx}/{len(supercells)}] Generating {actual_structures} structures for {sc_formula} ({num_batches} batches)")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            # Collect generated CIF files
            cif_files = list(supercell_dir.glob("*.cif"))
            
            if cif_files:
                all_cif_files.extend(cif_files)
                n_generated = len(cif_files)
                print(f"        Generated {n_generated} structures for {sc_formula}")
                total_generated += n_generated
                
                # Collect extxyz entries if file exists
                extxyz_file = supercell_dir / "generated_crystals.extxyz"
                if extxyz_file.exists():
                    with open(extxyz_file, 'r') as f:
                        all_extxyz_entries.append(f.read())
            else:
                print(f"        Failed to generate structures for {sc_formula}")
                all_success = False
                
        except subprocess.TimeoutExpired:
            print(f"        Timeout for {sc_formula} after {timeout} seconds")
            all_success = False
        except Exception as e:
            print(f"        Exception for {sc_formula}: {e}")
            all_success = False
    
    # Combine all generated files
    if all_cif_files:
        print(f"  Combining {len(all_cif_files)} structures from {len(supercells)} supercells...")
        
        # Create combined zip file
        combined_zip = output_dir / "generated_crystals_cif.zip"
        with zipfile.ZipFile(combined_zip, 'w') as zf:
            for cif_file in all_cif_files:
                zf.write(cif_file, arcname=cif_file.name)
        
        # Create combined extxyz file
        if all_extxyz_entries:
            combined_extxyz = output_dir / "generated_crystals.extxyz"
            with open(combined_extxyz, 'w') as f:
                f.write('\n'.join(all_extxyz_entries))
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"  Success! Generated {total_generated} total structures across {len(supercells)} supercells")
        print(f"  Output: {combined_zip.name}, {combined_extxyz.name if all_extxyz_entries else '(no extxyz)'}")
        return True
    else:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"  Failed: No structures generated")
        return False


def process_compositions(
    compositions_file: Path,
    output_base_dir: Path,
    model_path: str,
    n_structures: int,
    max_atoms: int = 20,
    max_compositions: int = -1,
    start_index: int = 0,
    skip_existing: bool = True
) -> Dict:
    """
    Process multiple compositions and generate structures.
    
    Args:
        compositions_file: JSON file with composition data
        output_base_dir: Base directory for all outputs
        model_path: MatterGen model path
        n_structures: Structures to generate per composition
        max_atoms: Maximum atoms per cell (for supercell expansion)
        max_compositions: Max compositions to process (-1 for all)
        start_index: Starting index in compositions list
        skip_existing: Skip compositions that already have generated structures
        
    Returns:
        Dictionary with generation statistics
    """
    # Load compositions
    with open(compositions_file) as f:
        compositions = json.load(f)
    
    # Apply slicing
    end_index = start_index + max_compositions if max_compositions > 0 else len(compositions)
    compositions = compositions[start_index:end_index]
    
    # Determine model type
    is_checkpoint = ('/' in model_path or Path(model_path).exists())
    model_type = "Fine-tuned checkpoint" if is_checkpoint else "Pretrained model"
    
    print("="*70)
    print(f"BATCH STRUCTURE GENERATION (CSP MODE)")
    print("="*70)
    print(f"Compositions file: {compositions_file}")
    print(f"Processing: {len(compositions)} compositions")
    print(f"Range: {start_index} to {end_index}")
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Structures per composition: {n_structures}")
    print(f"Max atoms per cell: {max_atoms} (supercell expansion enabled)")
    print(f"Batch size: Auto-calculated per composition")
    print(f"Resume mode: {'Enabled (skip existing)' if skip_existing else 'Disabled (regenerate all)'}")
    print("="*70)
    
    success_count = 0
    skipped_count = 0
    failed_compositions = []
    
    for idx, comp_data in enumerate(compositions, start=1):
        formula = comp_data['formula']
        composition = comp_data['composition']
        
        print(f"\n[{idx}/{len(compositions)}] Generating: {formula}")
        print(f"  Composition: {composition}")
        print(f"  Excess electrons: {comp_data['excess_electrons']:.2f}")
        print(f"  Total atoms: {comp_data['total_atoms']}")
        
        # Output directory for this composition
        output_dir = output_base_dir / f"{formula}_structures"
        
        # Check if already exists before attempting generation
        if skip_existing and check_if_already_generated(output_dir, min_structures=1):
            print(f"    Already generated (skipping)")
            skipped_count += 1
            success_count += 1
            continue
        
        # Generate structures
        success = generate_structures_for_composition(
            composition=composition,
            formula=formula,
            output_dir=output_dir,
            model_path=model_path,
            n_structures=n_structures,
            max_atoms=max_atoms,
            skip_if_exists=False  # Already checked above
        )
        
        if success:
            success_count += 1
        else:
            failed_compositions.append(formula)
        
        # Progress update every 10 compositions
        if idx % 10 == 0:
            print(f"\n--- Progress: {idx}/{len(compositions)} ({success_count} successful, {skipped_count} skipped) ---")
    
    # Generate statistics
    stats = {
        "total_processed": len(compositions),
        "successful": success_count,
        "skipped": skipped_count,
        "newly_generated": success_count - skipped_count,
        "failed": len(failed_compositions),
        "success_rate": success_count / len(compositions) * 100 if compositions else 0,
        "failed_compositions": failed_compositions
    }
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Total compositions processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    if skipped_count > 0:
        print(f"  - Skipped (already existed): {skipped_count}")
        print(f"  - Newly generated: {stats['newly_generated']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    if failed_compositions:
        print("\nFailed compositions:")
        for formula in failed_compositions[:20]:
            print(f"  - {formula}")
        if len(failed_compositions) > 20:
            print(f"  ... and {len(failed_compositions)-20} more")
        
        # Save failed compositions
        failed_file = output_base_dir / "failed_compositions.txt"
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed_compositions))
        print(f"\nFailed compositions saved to: {failed_file}")
    
    print("="*70)
    
    return stats


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate crystal structures for multiple compositions using MatterGen"
    )
    parser.add_argument(
        "--compositions", "-c",
        type=str,
        required=True,
        help="JSON file with compositions"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Base output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mattergen_base",
        help="MatterGen model: checkpoint path (e.g., outputs/singlerun/2025-10-10/15-30-45) or pretrained name (e.g., mattergen_base)"
    )
    parser.add_argument(
        "--n-structures", "-n",
        type=int,
        default=20,
        help="Number of structures per composition (default: 20)"
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=20,
        help="Maximum atoms per cell for supercell expansion (default: 20, matching mp_20 dataset)"
    )
    parser.add_argument(
        "--max-compositions", "-m",
        type=int,
        default=-1,
        help="Maximum compositions to process, -1 for all (default: -1)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in compositions list (default: 0)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip compositions that already have generated structures (default: True, enables resume)"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Regenerate all structures, even if they already exist"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    compositions_file = Path(args.compositions)
    output_base_dir = Path(args.output_dir)
    
    # Validate inputs
    if not compositions_file.exists():
        print(f"ERROR: Compositions file not found: {compositions_file}")
        sys.exit(1)
    
    # Run generation
    stats = process_compositions(
        compositions_file=compositions_file,
        output_base_dir=output_base_dir,
        model_path=args.model,
        n_structures=args.n_structures,
        max_atoms=args.max_atoms,
        max_compositions=args.max_compositions,
        start_index=args.start_index,
        skip_existing=args.skip_existing
    )
    
    # Save statistics
    stats_file = output_base_dir / "generation_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    main()

