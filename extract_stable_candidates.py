#!/usr/bin/env python3
"""
Extract stable candidates from hull_stability.json
"""

import json
import argparse
import zipfile
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Extract stable electride candidates from hull stability analysis"
    )
    parser.add_argument(
        '--hull-json',
        type=str,
        required=True,
        help="Path to hull_stability.json"
    )
    parser.add_argument(
        '--structures-zip',
        type=str,
        required=False,
        help="Path to generated_crystals_cif.zip (to extract CIF files)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='stable_candidates',
        help="Output directory for stable structures"
    )
    parser.add_argument(
        '--max-hull',
        type=float,
        default=0.01,
        help="Maximum E_hull (eV/atom) for stability (default: 0.01)"
    )
    parser.add_argument(
        '--sort-by',
        choices=['hull', 'energy', 'formula'],
        default='hull',
        help="Sort stable structures by (default: hull)"
    )
    
    args = parser.parse_args()
    
    # Load hull stability data
    print(f"Loading: {args.hull_json}")
    with open(args.hull_json, 'r') as f:
        hull_data = json.load(f)
    
    # Filter stable structures
    stable = [d for d in hull_data if d.get('is_stable', False)]
    
    # Apply E_hull threshold
    if args.max_hull != 0.01:
        stable = [d for d in stable if d['energy_above_hull'] <= args.max_hull]
    
    print(f"\nTotal structures analyzed: {len(hull_data)}")
    print(f"Stable structures (E_hull ≤ {args.max_hull} eV/atom): {len(stable)}")
    
    if not stable:
        print("\nNo stable structures found!")
        return
    
    # Sort
    if args.sort_by == 'hull':
        stable.sort(key=lambda x: x['energy_above_hull'])
    elif args.sort_by == 'energy':
        stable.sort(key=lambda x: x['energy_per_atom'])
    elif args.sort_by == 'formula':
        stable.sort(key=lambda x: x['formula'])
    
    # Group by chemical system
    by_chemsys = defaultdict(list)
    for s in stable:
        chemsys = s.get('chemsys', 'unknown')
        by_chemsys[chemsys].append(s)
    
    # Print summary
    print(f"\n{'='*80}")
    print("STABLE ELECTRIDE CANDIDATES")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Formula':<15} {'Chemical System':<15} {'E_hull (eV/atom)':<20} {'CIF File':<20}")
    print(f"{'-'*80}")
    
    for idx, s in enumerate(stable, 1):
        formula = s['formula']
        chemsys = s.get('chemsys', 'N/A')
        e_hull = s['energy_above_hull']
        cif_file = s['cif_file']
        
        print(f"{idx:<6} {formula:<15} {chemsys:<15} {e_hull:<20.6f} {cif_file:<20}")
    
    print(f"{'='*80}")
    
    # Summary by chemical system
    print(f"\n{'='*80}")
    print("SUMMARY BY CHEMICAL SYSTEM")
    print(f"{'='*80}")
    
    for chemsys in sorted(by_chemsys.keys()):
        structures = by_chemsys[chemsys]
        print(f"\n{chemsys}: {len(structures)} stable structure(s)")
        for s in structures[:5]:  # Show first 5
            print(f"  • {s['formula']}: E_hull = {s['energy_above_hull']:.6f} eV/atom ({s['cif_file']})")
        if len(structures) > 5:
            print(f"  ... and {len(structures)-5} more")
    
    # Save stable candidates list
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stable_json = output_dir / "stable_candidates.json"
    with open(stable_json, 'w') as f:
        json.dump(stable, f, indent=2)
    print(f"\nStable candidates saved to: {stable_json}")
    
    # Save by chemical system
    chemsys_json = output_dir / "by_chemical_system.json"
    with open(chemsys_json, 'w') as f:
        json.dump(dict(by_chemsys), f, indent=2)
    print(f"By chemical system: {chemsys_json}")
    
    # Extract CIF files if provided
    if args.structures_zip:
        print(f"\nExtracting CIF files from: {args.structures_zip}")
        cif_output_dir = output_dir / "cif_files"
        cif_output_dir.mkdir(parents=True, exist_ok=True)
        
        stable_cifs = {s['cif_file'] for s in stable}
        
        with zipfile.ZipFile(args.structures_zip, 'r') as zf:
            for cif_file in stable_cifs:
                try:
                    zf.extract(cif_file, cif_output_dir)
                    print(f"    Extracted: {cif_file}")
                except KeyError:
                    print(f"    Not found: {cif_file}")
        
        print(f"\nCIF files extracted to: {cif_output_dir}")
    
    # Create summary CSV
    import csv
    csv_file = output_dir / "stable_candidates.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rank', 'cif_file', 'formula', 'chemsys', 
            'energy_per_atom', 'energy_above_hull', 'decomposition'
        ])
        writer.writeheader()
        
        for idx, s in enumerate(stable, 1):
            writer.writerow({
                'rank': idx,
                'cif_file': s['cif_file'],
                'formula': s['formula'],
                'chemsys': s.get('chemsys', 'N/A'),
                'energy_per_atom': s['energy_per_atom'],
                'energy_above_hull': s['energy_above_hull'],
                'decomposition': str(s['decomposition'])
            })
    
    print(f"CSV file: {csv_file}")
    
    print(f"\n{'='*80}")
    print(f"Done! Found {len(stable)} stable electride candidates")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

