#!/usr/bin/env python3
"""
Convert extxyz files to CIF format using ASE
"""

import argparse
from pathlib import Path
from ase.io import read, write
import zipfile


def convert_extxyz_to_cif(extxyz_file: Path, output_dir: Path):
    """
    Convert a generated_crystals.extxyz file to individual CIF files and zip them.
    
    Args:
        extxyz_file: Path to the extxyz file
        output_dir: Directory containing the extxyz file (for output)
    """
    print(f"Converting {extxyz_file.name}...")
    
    # Read all structures from extxyz
    try:
        structures = read(str(extxyz_file), index=':')
        if not isinstance(structures, list):
            structures = [structures]
        
        print(f"  Found {len(structures)} structures")
        
        # Create temporary directory for CIF files
        temp_cif_dir = output_dir / ".temp_cif_conversion"
        temp_cif_dir.mkdir(exist_ok=True)
        
        # Convert each structure to CIF
        cif_files = []
        for i, structure in enumerate(structures):
            cif_file = temp_cif_dir / f"gen_{i}.cif"
            write(str(cif_file), structure, format='cif')
            cif_files.append(cif_file)
        
        # Create zip file
        zip_file = output_dir / "generated_crystals_cif.zip"
        with zipfile.ZipFile(zip_file, 'w') as zf:
            for cif_file in cif_files:
                zf.write(cif_file, arcname=cif_file.name)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_cif_dir)
        
        print(f"  Created {zip_file.name} with {len(cif_files)} CIF files")
        return True
        
    except Exception as e:
        print(f"  ERROR: Failed to convert: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert extxyz files to CIF format')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Results directory to scan (e.g., results/ternary_csp_magnets)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        return
    
    # Find all composition directories with extxyz but no zip
    total = 0
    converted = 0
    skipped = 0
    
    for comp_dir in sorted(results_dir.glob("*_structures")):
        if not comp_dir.is_dir():
            continue
        
        total += 1
        extxyz_file = comp_dir / "generated_crystals.extxyz"
        zip_file = comp_dir / "generated_crystals_cif.zip"
        
        if extxyz_file.exists() and not zip_file.exists():
            print(f"\n{comp_dir.name}:")
            if convert_extxyz_to_cif(extxyz_file, comp_dir):
                converted += 1
            else:
                print(f"  Failed to convert")
        else:
            skipped += 1
    
    print(f"\n{'='*70}")
    print(f"Conversion Summary:")
    print(f"  Total compositions: {total}")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already had zip or no extxyz): {skipped}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

