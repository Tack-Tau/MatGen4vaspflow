#!/usr/bin/env python3
"""
Test script to verify extxyz extraction and combination logic.
Uses the EXACT same logic as generate_structures_batch.py.
"""

from pathlib import Path
from ase.io import read


def extract_and_combine_extxyz(temp_dir: Path, output_dir: Path):
    """
    Extract extxyz files from supercell directories and combine them.
    This uses the EXACT same logic as generate_structures_batch.py.
    """
    all_extxyz_entries = []
    supercell_info = []
    
    supercell_dirs = sorted(temp_dir.glob("supercell_*"))
    
    for supercell_dir in supercell_dirs:
        print(f"\nProcessing {supercell_dir.name}:")
        
        # Check for extxyz file
        extxyz_file = supercell_dir / "generated_crystals.extxyz"
        
        if extxyz_file.exists():
            with open(extxyz_file, 'r') as f:
                extxyz_content = f.read()
            
            # Count structures in this extxyz
            lines = extxyz_content.split('\n')
            n_structures = 0
            for i, line in enumerate(lines):
                if line.strip() and line.strip()[0].isdigit():
                    if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                        n_structures += 1
            
            all_extxyz_entries.append(extxyz_content)
            supercell_info.append({
                'name': supercell_dir.name,
                'structures': n_structures,
                'size_bytes': len(extxyz_content)
            })
            
            print(f"  Found extxyz file: {n_structures} structures ({len(extxyz_content)} bytes)")
        else:
            print(f"  No extxyz file found")
    
    # Create combined extxyz file
    if all_extxyz_entries:
        total_structures = sum(info['structures'] for info in supercell_info)
        print(f"\nCombining {total_structures} structures from {len(supercell_dirs)} supercells...")
        
        combined_extxyz = output_dir / "generated_crystals.extxyz"
        with open(combined_extxyz, 'w') as f:
            # Concatenate extxyz files directly (no blank lines between)
            # Each entry is already a complete extxyz file from one supercell
            for content in all_extxyz_entries:
                # Ensure content ends with exactly one newline
                if not content.endswith('\n'):
                    content += '\n'
                f.write(content)
        
        print(f"Success! Created {combined_extxyz.name}")
        return True, combined_extxyz, supercell_info
    
    print("\nERROR: No extxyz files found")
    return False, None, []


def verify_extxyz(combined_extxyz: Path, expected_structures: int):
    """
    Verify that the combined extxyz file is correct.
    """
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    results = {}
    
    # Method 1: ASE read
    print(f"\nReading with ASE...")
    structures = read(str(combined_extxyz), index=':')
    if not isinstance(structures, list):
        structures = [structures]
    results['ase_count'] = len(structures)
    print(f"  ASE read: {results['ase_count']} structures")
    
    # Method 2: Manual counting (same as generate_structures_batch.py)
    print(f"\nManual line counting...")
    with open(combined_extxyz, 'r') as f:
        lines = f.read().split('\n')
    
    n_structures = 0
    structure_sizes = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and line[0].isdigit():
            if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                n_structures += 1
                try:
                    n_atoms = int(line.split()[0])
                    structure_sizes.append(n_atoms)
                except ValueError:
                    pass
        i += 1
    
    results['manual_count'] = n_structures
    print(f"  Manual count: {results['manual_count']} structures")
    
    # Method 3: Analyze file structure
    print(f"\nFile structure analysis...")
    file_size = combined_extxyz.stat().st_size
    print(f"  File size: {file_size} bytes ({file_size/1024:.1f} KB)")
    print(f"  Total lines: {len(lines)}")
    print(f"  Structure sizes (atoms): min={min(structure_sizes)}, max={max(structure_sizes)}, avg={sum(structure_sizes)/len(structure_sizes):.1f}")
    
    # Check for common issues
    print(f"\nChecking for common issues...")
    
    # Check for double blank lines (incorrect concatenation)
    double_blanks = 0
    for i in range(len(lines)-1):
        if not lines[i].strip() and not lines[i+1].strip():
            double_blanks += 1
    
    if double_blanks > 0:
        print(f"  ✗ WARNING: Found {double_blanks} double blank lines (may cause parsing issues)")
    else:
        print(f"  ✓ No double blank lines found")
    
    # Check for proper structure separation
    print(f"\nVerifying structure boundaries...")
    boundary_checks = []
    for i in range(len(lines)-2):
        if lines[i].strip() and lines[i].strip()[0].isdigit():
            if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                boundary_checks.append('valid')
    
    print(f"  Valid structure boundaries: {len(boundary_checks)}/{results['manual_count']}")
    
    # Show sample structures
    print(f"\nSample structures (first 3):")
    for i in range(min(3, len(structures))):
        struct = structures[i]
        print(f"  Structure {i}:")
        print(f"    Formula: {struct.get_chemical_formula()}")
        print(f"    Atoms: {len(struct)}")
        print(f"    Volume: {struct.get_volume():.2f} ų")
    
    # Compare with expected
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH EXPECTED")
    print(f"{'='*70}")
    print(f"  Expected structures: {expected_structures}")
    print(f"  ASE read: {results['ase_count']}")
    print(f"  Manual count: {results['manual_count']}")
    
    results['expected'] = expected_structures
    results['all_match'] = (
        results['ase_count'] == results['manual_count'] == expected_structures
    )
    
    if results['all_match']:
        print(f"\n  ✓ All counts match!")
    else:
        print(f"\n  ✗ Mismatch detected:")
        if results['ase_count'] != expected_structures:
            print(f"    - ASE read {results['ase_count']} but expected {expected_structures}")
        if results['manual_count'] != expected_structures:
            print(f"    - Manual count {results['manual_count']} but expected {expected_structures}")
    
    print(f"{'='*70}")
    return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_extxyz_extraction.py <composition_dir>")
        print("Example: python test_extxyz_extraction.py results/binary_csp_electrides/Mg3Br2_structures")
        sys.exit(1)
    
    comp_dir = Path(sys.argv[1])
    temp_dir = comp_dir / ".temp_generation"
    
    if not temp_dir.exists():
        print(f"ERROR: .temp_generation directory not found: {temp_dir}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"TESTING EXTXYZ EXTRACTION AND COMBINATION")
    print(f"{'='*70}")
    print(f"Composition directory: {comp_dir}")
    print(f"Temp directory: {temp_dir}")
    
    # Run extraction
    success, combined_extxyz, supercell_info = extract_and_combine_extxyz(temp_dir, comp_dir)
    
    if not success:
        print("\nERROR: Extraction failed!")
        sys.exit(1)
    
    # Calculate expected total
    expected_structures = sum(info['structures'] for info in supercell_info)
    
    print(f"\n{'='*70}")
    print(f"SUPERCELL BREAKDOWN")
    print(f"{'='*70}")
    for info in supercell_info:
        print(f"  {info['name']}:")
        print(f"    Structures: {info['structures']}")
        print(f"    Size: {info['size_bytes']} bytes ({info['size_bytes']/1024:.1f} KB)")
    print(f"\n  Total expected structures: {expected_structures}")
    
    # Verify output
    results = verify_extxyz(combined_extxyz, expected_structures)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    
    if results['all_match']:
        print(f"✓ ALL TESTS PASSED!")
        print(f"  - Extracted extxyz from {len(supercell_info)} supercells")
        print(f"  - Combined {results['ase_count']} structures correctly")
        print(f"  - ASE can read all structures")
        print(f"  - Manual counting matches ASE")
        print(f"\n✓ Extxyz concatenation logic is CORRECT")
    else:
        print(f"✗ SOME TESTS FAILED")
        print(f"  Please review the verification output above")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
