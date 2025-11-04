#!/usr/bin/env python
"""
Generate ternary electrides using fine-tuned is_electride model.

Since mattergen_base doesn't support CSP mode, we use the fine-tuned model
with is_electride conditioning to generate electride-like structures, then
filter by excess electron criteria 
"""

import subprocess
import json
from pathlib import Path
from collections import Counter

# Valence electrons dictionary
VALENCE_ELECTRONS = {
    # Group I
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    # Group II
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    # Group III
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'Sc': 3, 'Y': 3,
    # Lanthanides
    'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3,
    'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3,
    # Group IV
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    # Group V
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
    # Group VI
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'Po': 2,
    # Group VII
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'At': 1,
}

def generate_electrides_finetuned(
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 512,
    num_batches: int = 10,
    guidance_factor: float = 3.0
):
    """
    Generate electride structures using fine-tuned model.
    
    Args:
        output_dir: Output directory for structures
        checkpoint_path: Path to fine-tuned checkpoint 
        batch_size: Structures per batch
        num_batches: Number of batches
        guidance_factor: Guidance strength for is_electride conditioning
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mattergen-generate",
        str(output_dir),
        f"--model_path={checkpoint_path}",
        f"--batch_size={batch_size}",
        f"--num_batches={num_batches}",
        '--properties_to_condition_on={"is_electride": 1}',
        f"--diffusion_guidance_factor={guidance_factor}",
        "--record_trajectories=False"
    ]
    
    print("Generating electride structures with fine-tuned model...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Total structures: {batch_size * num_batches}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  Generation successful!")
    else:
        print(f"  Generation failed!")
        print(f"STDERR: {result.stderr}")
        return False
    
    return True


def calculate_excess_electrons(composition):
    """
    Calculate excess valence electrons
    
    Electronegative elements (N, O, P, S, etc.) act as electron acceptors.
    Electropositive elements donate electrons.
    Positive excess = potential electride.
    """
    electronegative = {'N', 'P', 'As', 'Sb', 'Bi', 
                       'O', 'S', 'Se', 'Te', 'Po',
                       'F', 'Cl', 'Br', 'I', 'At'}
    
    excess = 0.0
    
    for element, amount in composition.items():
        symbol = str(element)
        
        if symbol not in VALENCE_ELECTRONS:
            return None  # Unknown element
        
        valence = VALENCE_ELECTRONS[symbol]
        
        if symbol in electronegative:
            excess -= abs(valence) * amount
        else:
            excess += valence * amount
    
    return excess


def filter_by_excess_electrons(
    structures_zip: Path, 
    excess_range=(0.1, 4.0),
    ternary_only=True
):
    """
    Filter structures by excess electron criteria (not by pre-defined list).
    
    Args:
        structures_zip: Path to generated_crystals_cif.zip
        excess_range: (min, max) excess electrons for electride
        ternary_only: Only keep ternary compositions (3 unique elements)
    """
    import zipfile
    from pymatgen.io.cif import CifParser
    from io import StringIO
    
    print(f"\nFiltering structures by excess electron criteria...")
    print(f"Excess electron range: {excess_range[0]} to {excess_range[1]}")
    print(f"Ternary only: {ternary_only}")
    
    valid_structures = {}
    skipped_count = 0
    stats = {
        'total': 0,
        'ternary': 0,
        'valid_excess': 0,
        'final_matched': 0
    }
    
    with zipfile.ZipFile(structures_zip, 'r') as zf:
        cif_files = [f for f in zf.namelist() if f.endswith('.cif')]
        stats['total'] = len(cif_files)
        
        for cif_file in cif_files:
            with zf.open(cif_file) as f:
                cif_content = f.read().decode('utf-8')
                parser = CifParser(StringIO(cif_content))
                structure = parser.parse_structures(primitive=False)[0]
                
                composition = structure.composition.reduced_composition
                formula = composition.reduced_formula
                num_elements = len(composition.elements)
                
                # Check if ternary
                if ternary_only and num_elements != 3:
                    skipped_count += 1
                    continue
                
                if num_elements == 3:
                    stats['ternary'] += 1
                
                # Calculate excess electrons
                excess = calculate_excess_electrons(composition)
                
                if excess is None:
                    skipped_count += 1
                    continue
                
                # Check excess electron range
                if excess_range[0] < excess <= excess_range[1]:
                    stats['valid_excess'] += 1
                    
                    if formula not in valid_structures:
                        valid_structures[formula] = {
                            'cif_files': [],
                            'excess_electrons': excess,
                            'num_elements': num_elements,
                            'composition': {str(el): composition[el] for el in composition.elements}
                        }
                    
                    valid_structures[formula]['cif_files'].append(cif_file)
                    stats['final_matched'] += 1
    
    # Sort by excess electrons
    sorted_formulas = sorted(
        valid_structures.items(), 
        key=lambda x: x[1]['excess_electrons'], 
        reverse=True
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Filtering Statistics:")
    print(f"{'='*70}")
    print(f"  Total structures generated: {stats['total']}")
    print(f"  Ternary compositions: {stats['ternary']}")
    print(f"  Valid excess electrons: {stats['valid_excess']}")
    print(f"  Final matched: {stats['final_matched']}")
    print(f"  Unique compositions: {len(valid_structures)}")
    print(f"  Skipped: {skipped_count}")
    
    print(f"\n{'='*70}")
    print(f"Top 20 Compositions by Excess Electrons:")
    print(f"{'='*70}")
    for i, (formula, data) in enumerate(sorted_formulas[:20], 1):
        n_structs = len(data['cif_files'])
        excess = data['excess_electrons']
        print(f"  {i:2d}. {formula:15s} | {excess:5.2f} e⁻ | {n_structs:3d} structures")
    
    return valid_structures, stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ternary electrides using fine-tuned model with excess electron filtering"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--output_dir", type=str, 
                        default="../results/electrides_finetuned",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Structures per batch")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of batches")
    parser.add_argument("--guidance_factor", type=float, default=3.0,
                        help="Guidance factor for is_electride conditioning")
    parser.add_argument("--excess_min", type=float, default=0.1,
                        help="Minimum excess electrons for electride (default: 0.1)")
    parser.add_argument("--excess_max", type=float, default=4.0,
                        help="Maximum excess electrons for electride (default: 4.0)")
    parser.add_argument("--ternary_only", action="store_true", default=True,
                        help="Only keep ternary compositions (3 elements)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Ternary Electride Generation (Fine-tuned Model)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num batches: {args.num_batches}")
    print(f"Total structures: {args.batch_size * args.num_batches}")
    print(f"Excess electron range: {args.excess_min} - {args.excess_max}")
    print(f"Ternary only: {args.ternary_only}")
    print("="*70)
    
    # Generate structures
    success = generate_electrides_finetuned(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        guidance_factor=args.guidance_factor
    )
    
    if not success:
        print("Generation failed!")
        return
    
    # Filter by excess electrons
    structures_zip = Path(args.output_dir) / "generated_crystals_cif.zip"
    
    if structures_zip.exists():
        valid_structures, stats = filter_by_excess_electrons(
            structures_zip=structures_zip,
            excess_range=(args.excess_min, args.excess_max),
            ternary_only=args.ternary_only
        )
        
        # Save results
        output_json = Path(args.output_dir) / "electride_candidates.json"
        with open(output_json, 'w') as f:
            json.dump(valid_structures, f, indent=2)
        
        # Save stats
        stats_json = Path(args.output_dir) / "filtering_stats.json"
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved:")
        print(f"  Electride candidates: {output_json}")
        print(f"  Filtering statistics: {stats_json}")
        print(f"{'='*70}")
    else:
        print(f"  Generated structures not found at: {structures_zip}")


if __name__ == "__main__":
    main()

