#!/usr/bin/env python
"""
Filter generated structures by ternary composition and excess electrons.
Use this if generation completed but filtering failed.
"""

import json
import zipfile
from pathlib import Path
from io import StringIO
import argparse

# Valence electrons dictionary
VALENCE_ELECTRONS = {
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3, 'Sc': 3, 'Y': 3,
    'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3,
    'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3,
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'Po': 2,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'At': 1,
}

def calculate_excess_electrons(composition):
    """Calculate excess valence electrons."""
    electronegative = {'N', 'P', 'As', 'Sb', 'Bi', 
                       'O', 'S', 'Se', 'Te', 'Po',
                       'F', 'Cl', 'Br', 'I', 'At'}
    
    excess = 0.0
    
    for element, amount in composition.items():
        symbol = str(element)
        
        if symbol not in VALENCE_ELECTRONS:
            return None
        
        valence = VALENCE_ELECTRONS[symbol]
        
        if symbol in electronegative:
            excess -= abs(valence) * amount
        else:
            excess += valence * amount
    
    return excess


def filter_structures(structures_zip, excess_range=(0.1, 4.0), ternary_only=True):
    """Filter structures by excess electrons and ternary composition."""
    from pymatgen.io.cif import CifParser
    
    print(f"\nFiltering structures...")
    print(f"Excess electron range: {excess_range[0]} - {excess_range[1]}")
    print(f"Ternary only: {ternary_only}")
    
    valid_structures = {}
    stats = {
        'total': 0,
        'ternary': 0,
        'valid_excess': 0,
        'final_matched': 0
    }
    
    with zipfile.ZipFile(structures_zip, 'r') as zf:
        cif_files = [f for f in zf.namelist() if f.endswith('.cif')]
        stats['total'] = len(cif_files)
        
        print(f"Processing {len(cif_files)} structures...")
        
        for i, cif_file in enumerate(cif_files, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(cif_files)}...")
            
            try:
                with zf.open(cif_file) as f:
                    cif_content = f.read().decode('utf-8')
                    parser = CifParser(StringIO(cif_content))
                    structure = parser.parse_structures(primitive=True)[0]
                    
                    composition = structure.composition.reduced_composition
                    formula = composition.reduced_formula
                    num_elements = len(composition.elements)
                    
                    if ternary_only and num_elements != 3:
                        continue
                    
                    if num_elements == 3:
                        stats['ternary'] += 1
                    
                    excess = calculate_excess_electrons(composition)
                    
                    if excess is None:
                        continue
                    
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
            except Exception as e:
                print(f"  Warning: Failed to process {cif_file}: {e}")
                continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Filtering Results:")
    print(f"{'='*70}")
    print(f"  Total structures: {stats['total']}")
    print(f"  Ternary structures: {stats['ternary']}")
    print(f"  Valid excess electrons: {stats['valid_excess']}")
    print(f"  Final matched: {stats['final_matched']}")
    print(f"  Unique compositions: {len(valid_structures)}")
    
    # Top 20
    sorted_formulas = sorted(
        valid_structures.items(),
        key=lambda x: x[1]['excess_electrons'],
        reverse=True
    )
    
    print(f"\nTop 20 Compositions by Excess Electrons:")
    print(f"{'='*70}")
    for i, (formula, data) in enumerate(sorted_formulas[:20], 1):
        n = len(data['cif_files'])
        e = data['excess_electrons']
        print(f"  {i:2d}. {formula:15s} | {e:5.2f} e⁻ | {n:3d} structures")
    
    return valid_structures, stats


def main():
    parser = argparse.ArgumentParser(description="Filter ternary electride candidates")
    parser.add_argument("--structures_zip", type=str, required=True,
                        help="Path to generated_crystals_cif.zip")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--excess_min", type=float, default=0.1)
    parser.add_argument("--excess_max", type=float, default=4.0)
    parser.add_argument("--ternary_only", action="store_true", default=True)
    
    args = parser.parse_args()
    
    structures_zip = Path(args.structures_zip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not structures_zip.exists():
        print(f"Error: {structures_zip} not found!")
        return 1
    
    print("="*70)
    print("Filtering Ternary Electrides")
    print("="*70)
    print(f"Input: {structures_zip}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Filter
    valid_structures, stats = filter_structures(
        structures_zip=structures_zip,
        excess_range=(args.excess_min, args.excess_max),
        ternary_only=args.ternary_only
    )
    
    # Save results
    candidates_file = output_dir / "electride_candidates.json"
    with open(candidates_file, 'w') as f:
        json.dump(valid_structures, f, indent=2)
    
    stats_file = output_dir / "filtering_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved:")
    print(f"  {candidates_file}")
    print(f"  {stats_file}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit(main())

