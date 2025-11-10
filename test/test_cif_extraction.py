#!/usr/bin/env python3
"""
Test script to verify CIF extraction and renaming logic.
Uses the exact same logic as generate_structures_batch.py.
"""

from pathlib import Path
import zipfile
from ase.io import read


def extract_and_combine_cifs(temp_dir: Path, output_dir: Path):
    """
    Extract CIF files from supercell zip archives and combine them.
    This uses the EXACT same logic as generate_structures_batch.py.
    """
    all_cif_files = []
    all_extxyz_entries = []
    total_generated = 0

    supercell_dirs = sorted(temp_dir.glob("supercell_*"))

    for supercell_idx, supercell_dir in enumerate(supercell_dirs, 1):
        print(f"\nProcessing {supercell_dir.name}:")

        # Collect generated CIF files (check both individual files and zip)
        cif_files = list(supercell_dir.glob("*.cif"))
        zip_file = supercell_dir / "generated_crystals_cif.zip"

        # If no individual CIF files but zip exists, extract and rename them
        if not cif_files and zip_file.exists():
            temp_cif_extract = supercell_dir / ".temp_cif_extract"
            temp_cif_extract.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.cif'):
                        # Extract with renamed file (prefix with supercell name)
                        new_name = f"{supercell_dir.name}_{name}"
                        extract_path = temp_cif_extract / new_name
                        with open(extract_path, 'wb') as f:
                            f.write(zf.read(name))
                        cif_files.append(extract_path)

            print(f"  Extracted {len(cif_files)} CIF files from zip")
            print(f"  Sample names: {[f.name for f in cif_files[:3]]}")

        # Check for extxyz file as alternative success indicator
        extxyz_file = supercell_dir / "generated_crystals.extxyz"
        extxyz_exists = extxyz_file.exists()

        # Success if either CIF files or extxyz file exists
        if cif_files or extxyz_exists:
            if cif_files:
                all_cif_files.extend(cif_files)
                n_generated = len(cif_files)
                print(f"  Generated {n_generated} CIF structures")
                total_generated += n_generated

            # Collect extxyz entries if file exists
            if extxyz_exists:
                with open(extxyz_file, 'r') as f:
                    extxyz_content = f.read()
                    all_extxyz_entries.append(extxyz_content)
                    # Count structures in extxyz
                    if not cif_files:
                        lines = extxyz_content.split('\n')
                        n_structures = 0
                        for i, line in enumerate(lines):
                            if line.strip() and line.strip()[0].isdigit():
                                if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                                    n_structures += 1
                        print(f"  Generated {n_structures} structures (extxyz only)")
                        total_generated += n_structures

    # Combine all generated files
    if all_cif_files or all_extxyz_entries:
        if all_cif_files:
            print(f"\nCombining {len(all_cif_files)} CIF structures from {len(supercell_dirs)} supercells...")

            # Create combined zip file
            combined_zip = output_dir / "generated_crystals_cif.zip"
            with zipfile.ZipFile(combined_zip, 'w') as zf:
                for cif_file in all_cif_files:
                    zf.write(cif_file, arcname=cif_file.name)

        # Create combined extxyz file
        if all_extxyz_entries:
            if not all_cif_files:
                print(f"\nCombining {total_generated} structures from {len(supercell_dirs)} supercells (extxyz only)...")
            combined_extxyz = output_dir / "generated_crystals.extxyz"
            with open(combined_extxyz, 'w') as f:
                # Concatenate extxyz files directly (no blank lines between)
                for content in all_extxyz_entries:
                    if not content.endswith('\n'):
                        content += '\n'
                    f.write(content)

        print(f"\nSuccess! Generated {total_generated} total structures across {len(supercell_dirs)} supercells")
        if all_cif_files and all_extxyz_entries:
            print(f"Output: {combined_zip.name}, {combined_extxyz.name}")
        elif all_cif_files:
            print(f"Output: {combined_zip.name} (no extxyz)")
        else:
            print(f"Output: {combined_extxyz.name} (extxyz only)")

        return True, combined_zip if all_cif_files else None, combined_extxyz if all_extxyz_entries else None

    return False, None, None


def verify_outputs(combined_zip: Path, combined_extxyz: Path):
    """
    Verify that the combined outputs are correct.
    """
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    results = {}

    # Verify combined ZIP
    if combined_zip and combined_zip.exists():
        with zipfile.ZipFile(combined_zip, 'r') as zf:
            cif_names = [n for n in zf.namelist() if n.endswith('.cif')]
            results['zip_total'] = len(cif_names)
            results['zip_unique'] = len(set(cif_names))

            print(f"\nCombined ZIP: {combined_zip.name}")
            print(f"  Total CIF files: {results['zip_total']}")
            print(f"  Unique names: {results['zip_unique']}")

            if results['zip_total'] == results['zip_unique']:
                print(f"  ✓ All CIF names are unique")
            else:
                print(f"  ✗ WARNING: Duplicate CIF names found!")
                duplicates = [name for name in cif_names if cif_names.count(name) > 1]
                print(f"  Duplicates: {set(duplicates)}")

            # Show distribution by supercell
            supercell_counts = {}
            for name in cif_names:
                supercell = name.split('_gen_')[0]
                supercell_counts[supercell] = supercell_counts.get(supercell, 0) + 1

            print(f"  Distribution by supercell:")
            for sc, count in sorted(supercell_counts.items()):
                print(f"    {sc}: {count} files")

            results['supercell_counts'] = supercell_counts

    # Verify combined extxyz
    if combined_extxyz and combined_extxyz.exists():
        # Method 1: ASE read
        structures = read(str(combined_extxyz), index=':')
        if not isinstance(structures, list):
            structures = [structures]
        results['ase_count'] = len(structures)

        # Method 2: Manual counting (same as generate_structures_batch.py)
        with open(combined_extxyz, 'r') as f:
            lines = f.read().split('\n')
        n_structures = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                    n_structures += 1
        results['manual_count'] = n_structures

        print(f"\nCombined EXTXYZ: {combined_extxyz.name}")
        print(f"  ASE read: {results['ase_count']} structures")
        print(f"  Manual count: {results['manual_count']} structures")

        if results['ase_count'] == results['manual_count']:
            print(f"  ✓ ASE and manual counts match")
        else:
            print(f"  ✗ WARNING: Count mismatch!")

    # Cross-check CIF vs extxyz counts
    if 'zip_total' in results and 'ase_count' in results:
        print(f"\nCross-check:")
        print(f"  CIF files in zip: {results['zip_total']}")
        print(f"  Structures in extxyz: {results['ase_count']}")

        if results['zip_total'] == results['ase_count']:
            print(f"  ✓ CIF and extxyz counts match")
        else:
            print(f"  ✗ WARNING: Counts do not match!")

    print(f"\n{'='*70}")
    return results


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_cif_extraction.py <composition_dir>")
        print("Example: python test_cif_extraction.py results/binary_csp_electrides/Mg3Br2_structures")
        sys.exit(1)

    comp_dir = Path(sys.argv[1])
    temp_dir = comp_dir / ".temp_generation"

    if not temp_dir.exists():
        print(f"ERROR: .temp_generation directory not found: {temp_dir}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"TESTING CIF EXTRACTION AND COMBINATION")
    print(f"{'='*70}")
    print(f"Composition directory: {comp_dir}")
    print(f"Temp directory: {temp_dir}")

    # Run extraction
    success, combined_zip, combined_extxyz = extract_and_combine_cifs(temp_dir, comp_dir)

    if not success:
        print("\nERROR: Extraction failed!")
        sys.exit(1)

    # Verify outputs
    results = verify_outputs(combined_zip, combined_extxyz)

    # Final summary
    print("\nFINAL SUMMARY:")
    if combined_zip and combined_extxyz:
        print(f"  ✓ Both generated_crystals_cif.zip and generated_crystals.extxyz created")
    elif combined_zip:
        print(f"  ✓ Only generated_crystals_cif.zip created (no extxyz)")
    elif combined_extxyz:
        print(f"  ✓ Only generated_crystals.extxyz created (no CIF zip)")

    if all([
        results.get('zip_total') == results.get('zip_unique'),
        results.get('ase_count') == results.get('manual_count'),
        results.get('zip_total') == results.get('ase_count')
    ]):
        print(f"\n{'='*70}")
        print(f"✓ ALL TESTS PASSED!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"✗ SOME TESTS FAILED - Please review the output above")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

