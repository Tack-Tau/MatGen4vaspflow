#!/usr/bin/env python
"""
Extract structures for specific compositions from generated results.

After running generate_with_finetuned.py, use this script to extract
structures for specific compositions of interest.
"""

import json
import zipfile
import shutil
from pathlib import Path
import argparse

def extract_structures(
    structures_zip: Path,
    electride_candidates: Path,
    target_compositions: list,
    output_dir: Path,
    min_structures: int = 5
):
    """
    Extract CIF files for target compositions from electride candidates.
    
    Args:
        structures_zip: Path to generated_crystals_cif.zip
        electride_candidates: Path to electride_candidates.json
        target_compositions: List of composition formulas to extract (or "all")
        output_dir: Output directory for extracted structures
        min_structures: Minimum structures required for a composition
    """
    
    # Load electride candidates
    with open(electride_candidates, 'r') as f:
        candidates = json.load(f)
    
    # Determine which compositions to extract
    if target_compositions == ["all"]:
        # Extract all compositions with >= min_structures
        to_extract = {
            comp: data for comp, data in candidates.items() 
            if len(data['cif_files']) >= min_structures
        }
    else:
        # Extract specific compositions
        to_extract = {
            comp: candidates.get(comp, {'cif_files': [], 'excess_electrons': 0}) 
            for comp in target_compositions
        }
    
    if not to_extract:
        print("No compositions to extract!")
        return
    
    print(f"Extracting structures for {len(to_extract)} compositions...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract structures
    total_extracted = 0
    
    with zipfile.ZipFile(structures_zip, 'r') as zf:
        for comp, data in to_extract.items():
            cif_files = data.get('cif_files', [])
            excess_e = data.get('excess_electrons', 0)
            
            if not cif_files:
                print(f"  {comp}: No structures found")
                continue
            
            # Create composition directory
            comp_dir = output_dir / comp
            comp_dir.mkdir(exist_ok=True)
            
            # Extract CIF files
            for cif_file in cif_files:
                output_path = comp_dir / Path(cif_file).name
                
                with zf.open(cif_file) as src:
                    with open(output_path, 'wb') as dst:
                        dst.write(src.read())
            
            total_extracted += len(cif_files)
            print(f"  {comp:15s} | {excess_e:5.2f} e⁻ | {len(cif_files):3d} structures -> {comp_dir}")
    
    print(f"\n  Extracted {total_extracted} structures to {output_dir}")
    
    # Create summary with excess electrons
    summary = {
        "total_compositions": len(to_extract),
        "total_structures": total_extracted,
        "compositions": {
            comp: {
                "count": len(data.get('cif_files', [])),
                "excess_electrons": data.get('excess_electrons', 0)
            } for comp, data in to_extract.items()
        }
    }
    
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structures for specific compositions"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Results directory from generate_with_finetuned.py")
    parser.add_argument("--compositions", type=str, nargs='+', default=["all"],
                        help="Composition formulas to extract (e.g., Li3BN or 'all')")
    parser.add_argument("--min_structures", type=int, default=5,
                        help="Minimum structures required (only for 'all' mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results_dir/extracted)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    structures_zip = results_dir / "generated_crystals_cif.zip"
    electride_candidates = results_dir / "electride_candidates.json"
    
    if not structures_zip.exists():
        print(f"Error: {structures_zip} not found!")
        return
    
    if not electride_candidates.exists():
        print(f"Error: {electride_candidates} not found!")
        return
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = results_dir / "extracted"
    else:
        output_dir = Path(args.output_dir)
    
    print("="*70)
    print("Extracting Composition Structures")
    print("="*70)
    print(f"Results: {results_dir}")
    print(f"Target compositions: {args.compositions}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    extract_structures(
        structures_zip=structures_zip,
        electride_candidates=electride_candidates,
        target_compositions=args.compositions,
        output_dir=output_dir,
        min_structures=args.min_structures
    )


if __name__ == "__main__":
    main()

