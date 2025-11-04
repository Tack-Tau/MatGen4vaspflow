#!/usr/bin/env python3
"""
Evaluate thermodynamic stability using convex hull analysis.

ENERGY REFERENCE HANDLING:
======================================
This script ensures consistent energy references by re-relaxing ALL structures
(both your generated structures and MP competing phases) with MatterSim.

WORKFLOW:
1. Load your MatterSim-relaxed structures (from relaxed_structures.extxyz)
2. Download MP competing phase structures for the chemical system
3. Re-relax MP structures with MatterSim (cached to avoid re-computation)
4. Build convex hull using all MatterSim energies
5. Compute energy_above_hull on consistent energy scale

This provides fast thermodynamic screening. For publication-quality results,
re-run with VASP/DFT energies after initial screening.
"""

import json
import sys
import zipfile
import argparse
from pathlib import Path
from io import StringIO
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress harmless uncertainty warnings from MP API
warnings.filterwarnings('ignore', message='Using UFloat objects with std_dev==0')

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester

# MatterSim imports
try:
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    from mattersim.forcefield import MatterSimCalculator
    MATTERSIM_AVAILABLE = True
except ImportError:
    MATTERSIM_AVAILABLE = False
    print("WARNING: MatterSim not available - Option 2 requires MatterSim!")


def load_relaxed_structures_and_energies(extxyz_path, stability_json_path, max_structures=None):
    """
    Load relaxed structures from extxyz file and match with energies from stability.json.
    
    This is preferred over loading original CIF structures because we want to compute
    hull distance for the RELAXED structures, not the original MatterGen outputs.
    """
    from ase.io import read as ase_read
    from pymatgen.io.ase import AseAtomsAdaptor
    
    # Load stability data
    with open(stability_json_path, 'r') as f:
        stability_data = json.load(f)
    
    # Load relaxed structures from extxyz
    print(f"Loading relaxed structures from {extxyz_path}...")
    relaxed_atoms_list = ase_read(extxyz_path, index=':')
    
    if not isinstance(relaxed_atoms_list, list):
        relaxed_atoms_list = [relaxed_atoms_list]
    
    print(f"Loaded {len(relaxed_atoms_list)} relaxed structures")
    print(f"Found {len(stability_data)} entries in stability.json")
    
    # Match structures with stability data
    structures = []
    adaptor = AseAtomsAdaptor()
    
    for idx, (atoms, stab_entry) in enumerate(tqdm(
        zip(relaxed_atoms_list, stability_data), 
        desc="Processing structures",
        total=min(len(relaxed_atoms_list), len(stability_data))
    )):
        if max_structures and idx >= max_structures:
            break
            
        if not stab_entry.get('converged', False):
            continue
        
        try:
            # Convert ASE atoms to pymatgen Structure
            structure = adaptor.get_structure(atoms)
            
            structures.append({
                'cif_file': stab_entry['cif_file'],
                'structure': structure,
                'formula': structure.composition.reduced_formula,
                'composition': structure.composition,
                'energy_per_atom': stab_entry['energy_per_atom']
            })
        except Exception as e:
            print(f"  Warning: Could not process structure {idx}: {e}")
    
    print(f"Successfully processed {len(structures)} relaxed structures")
    return structures


def load_structures_from_zip(zip_path, max_structures=None):
    """
    Load structures from ZIP file (ORIGINAL structures, not relaxed).
    
    NOTE: For hull analysis, use load_relaxed_structures_and_energies() instead!
    This function is kept for backward compatibility only.
    """
    structures = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        cif_files = sorted([f for f in zf.namelist() if f.endswith('.cif')])
        
        if max_structures:
            cif_files = cif_files[:max_structures]
        
        print(f"Loading {len(cif_files)} structures...")
        
        for cif_file in tqdm(cif_files, desc="Parsing CIFs"):
            try:
                with zf.open(cif_file) as f:
                    cif_content = f.read().decode('utf-8')
                    parser = CifParser(StringIO(cif_content))
                    structure = parser.parse_structures(primitive=False)[0]
                    
                    structures.append({
                        'cif_file': cif_file,
                        'structure': structure,
                        'formula': structure.composition.reduced_formula,
                        'composition': structure.composition
                    })
            except Exception as e:
                print(f"  Warning: Could not parse {cif_file}: {e}")
    
    print(f"Successfully loaded {len(structures)} structures")
    return structures


def load_relaxed_energies(stability_json_path):
    """
    Load relaxed energies from stability.json.
    
    NOTE: This is deprecated when using load_relaxed_structures_and_energies()
    which loads both structures and energies together.
    """
    with open(stability_json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping: cif_file -> energy_per_atom
    energy_map = {}
    for entry in data:
        if entry.get('converged', False):
            cif_file = entry['cif_file']
            energy_per_atom = entry['energy_per_atom']
            energy_map[cif_file] = energy_per_atom
    
    return energy_map


def relax_structure_mattersim(structure, device='cpu', fmax=0.01, max_steps=500):
    """
    Relax a structure using MatterSim + FIRE optimizer.
    
    Matches the relaxation setup in relax_structures_manual.py for consistency.
    
    Args:
        structure: Pymatgen Structure
        device: 'cpu' or 'cuda'
        fmax: Force convergence criterion (eV/Å)
        max_steps: Maximum optimization steps
        
    Returns:
        tuple: (relaxed_structure, energy_per_atom)
    """
    if not MATTERSIM_AVAILABLE:
        raise ImportError("MatterSim not available!")
    
    # Convert to ASE Atoms
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    
    # Set up MatterSim calculator (same as relax_structures_manual.py)
    calc = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth",
        device=device
    )
    atoms.calc = calc
    
    # Relax with FIRE optimizer (more robust than BFGS)
    optimizer = FIRE(atoms, logfile=None)
    try:
        optimizer.run(fmax=fmax, steps=max_steps)
        energy = atoms.get_potential_energy()
        energy_per_atom = energy / len(atoms)
        
        # Convert back to pymatgen
        relaxed_structure = adaptor.get_structure(atoms)
        
        return relaxed_structure, energy_per_atom
    except Exception as e:
        print(f"    Warning: Relaxation failed: {e}")
        # Return original structure with high energy to mark as unstable
        return structure, 1e10


def get_competing_phases_from_mp_with_mattersim(
    composition, 
    mp_api_key, 
    cache_dir=None, 
    device='cpu',
    max_retries=3
):
    """
    Get competing phases from MP and re-relax them with MatterSim.
    
    This ensures all energies are on the same MatterSim reference scale.
    Results are cached to avoid re-computation.
    
    Args:
        composition: Pymatgen Composition object
        mp_api_key: Materials Project API key
        cache_dir: Directory to cache relaxed MP structures
        device: 'cpu' or 'cuda' for MatterSim
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of ComputedEntry objects with MatterSim energies
    """
    import time
    from requests.exceptions import HTTPError
    
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    elements = sorted([str(el) for el in composition.elements])
    chemsys = '-'.join(elements)
    
    # Check cache first
    cache_file = cache_dir / f"mp_{chemsys}_mattersim.json" if cache_dir else None
    if cache_file and cache_file.exists():
        print(f"    Loading cached MatterSim-relaxed MP phases for {chemsys}...")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Reconstruct ComputedEntry objects
        entries = []
        for item in cached_data:
            entry = ComputedEntry(
                composition=item['composition'],
                energy=item['energy'],
                entry_id=item['entry_id']
            )
            entries.append(entry)
        return entries
    
    # Download MP entries for entire chemical system
    print(f"    Downloading MP entries for {chemsys}...")
    for attempt in range(max_retries):
        try:
            with MPRester(mp_api_key) as mpr:
                # Get ALL entries in chemical system (includes elemental phases!)
                mp_entries = mpr.get_entries_in_chemsys(elements)
                
                if not mp_entries:
                    print(f"    Warning: No MP entries found for {chemsys}")
                    return []
                
                # Show breakdown by subsystem
                phase_count = {}
                for entry in mp_entries:
                    comp_els = tuple(sorted([str(el) for el in entry.composition.elements]))
                    phase_count[comp_els] = phase_count.get(comp_els, 0) + 1
                
                print(f"    Found {len(mp_entries)} MP entries across subsystems:")
                for comp_els, count in sorted(phase_count.items()):
                    sub_chemsys = '-'.join(comp_els)
                    print(f"      {sub_chemsys}: {count} phases")
                
                print(f"    Downloading structures and relaxing with MatterSim...")
                sys.stdout.flush()
                
                # Re-relax all MP structures with MatterSim
                mattersim_entries = []
                fetch_errors = 0
                relax_errors = 0
                
                for mp_entry in tqdm(mp_entries, desc=f"    Relaxing {chemsys}", leave=False):
                    # First, try to get structure from entry or fetch it
                    try:
                        if hasattr(mp_entry, 'structure') and mp_entry.structure is not None:
                            mp_struct = mp_entry.structure
                        else:
                            mp_result = mpr.get_structure_by_material_id(mp_entry.entry_id)
                            
                            if isinstance(mp_result, list):
                                if len(mp_result) == 0:
                                    fetch_errors += 1
                                    continue
                                mp_struct = mp_result[0]
                            else:
                                mp_struct = mp_result
                        
                        if not hasattr(mp_struct, 'composition'):
                            fetch_errors += 1
                            continue
                            
                    except Exception:
                        fetch_errors += 1
                        continue
                    
                    # Now try to relax the structure
                    try:
                        relaxed_struct, energy_per_atom = relax_structure_mattersim(
                            mp_struct, device=device
                        )
                        
                        if energy_per_atom > 1e9:
                            relax_errors += 1
                            continue
                        
                        total_energy = energy_per_atom * relaxed_struct.composition.num_atoms
                        
                        entry = ComputedEntry(
                            composition=relaxed_struct.composition,
                            energy=total_energy,
                            entry_id=f"mp_mattersim_{mp_entry.entry_id}"
                        )
                        mattersim_entries.append(entry)
                    except Exception:
                        relax_errors += 1
                
                # Show success/failure stats
                if fetch_errors > 0 or relax_errors > 0:
                    print(f"    Successfully relaxed {len(mattersim_entries)}/{len(mp_entries)} structures")
                    if fetch_errors > 0:
                        print(f"    Structure fetch errors: {fetch_errors}")
                    if relax_errors > 0:
                        print(f"    Relaxation errors: {relax_errors}")
                    sys.stdout.flush()
                
                # Cache results
                if cache_file and len(mattersim_entries) > 0:
                    cached_data = []
                    for entry in mattersim_entries:
                        cached_data.append({
                            'composition': entry.composition.as_dict(),
                            'energy': entry.energy,
                            'entry_id': entry.entry_id
                        })
                    with open(cache_file, 'w') as f:
                        json.dump(cached_data, f, indent=2)
                    print(f"    Cached {len(mattersim_entries)} MatterSim-relaxed MP phases")
                    sys.stdout.flush()
                
                if len(mattersim_entries) == 0:
                    print(f"    WARNING: Failed to relax all {len(mp_entries)} MP structures!")
                    print(f"    Hull analysis will fail for this chemical system.")
                    sys.stdout.flush()
                
                return mattersim_entries
                
        except HTTPError as e:
            if '429' in str(e):  # Rate limit error
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                print(f"    Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception(f"Failed to fetch MP data after {max_retries} attempts")


def compute_hull_distance(structure, energy_per_atom, mp_entries):
    """
    Compute energy above hull for a structure.
    
    Args:
        structure: Pymatgen Structure
        energy_per_atom: Energy per atom (eV/atom)
        mp_entries: List of ComputedEntry from MP
        
    Returns:
        dict with hull analysis results, or None if insufficient data
    """
    if not mp_entries:
        raise ValueError("No MP entries provided - cannot compute hull distance")
    
    composition = structure.composition
    
    # Create entry for this structure
    total_energy = energy_per_atom * composition.num_atoms
    
    entry = ComputedEntry(
        composition=composition,
        energy=total_energy,
        entry_id='generated'
    )
    
    # Add to entries
    all_entries = list(mp_entries) + [entry]
    
    # Build phase diagram - this may fail if terminal entries are missing
    try:
        pd = PhaseDiagram(all_entries)
    except Exception as e:
        raise ValueError(f"Failed to build phase diagram: {e}")
    
    # Get energy above hull
    e_above_hull = pd.get_e_above_hull(entry)
    
    # Get decomposition products
    decomp = pd.get_decomp_and_e_above_hull(entry)
    
    # Check if on hull
    is_stable = e_above_hull < 0.01  # eV/atom threshold
    
    return {
        'energy_above_hull': float(e_above_hull),
        'is_stable': bool(is_stable),  # Convert numpy bool_ to Python bool
        'decomposition': {k.reduced_formula: v for k, v in decomp[0].items()},
        'phase_diagram': pd
    }


def plot_phase_diagram(pd, entry, output_path, title="Phase Diagram"):
    """Plot phase diagram with the generated structure highlighted."""
    plotter = PDPlotter(pd, show_unstable=False)
    
    # Get the plot
    fig = plotter.get_plot()
    
    # Highlight our entry
    ax = fig.gca()
    
    # Find coordinates of our entry
    comp = entry.composition
    if len(comp.elements) == 2:
        # Binary system - can plot
        x_coord = comp.fractional_composition.get_atomic_fraction(comp.elements[0])
        e_above_hull = pd.get_e_above_hull(entry)
        
        ax.plot([x_coord], [e_above_hull], 'r*', markersize=15, 
                label='Generated structure')
        ax.legend()
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Phase diagram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stability using convex hull analysis"
    )
    parser.add_argument(
        '--structures-zip',
        type=str,
        required=False,
        default=None,
        help="Path to generated_crystals_cif.zip (ORIGINAL structures - not recommended)"
    )
    parser.add_argument(
        '--relaxed-extxyz',
        type=str,
        required=False,
        default=None,
        help="Path to relaxed_structures.extxyz (PREFERRED - uses relaxed structures)"
    )
    parser.add_argument(
        '--stability-json',
        type=str,
        required=True,
        help="Path to stability.json from relaxation"
    )
    parser.add_argument(
        '--mp-api-key',
        type=str,
        default=None,
        help="Materials Project API key (or set MP_API_KEY env variable)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory (default: same as stability-json)"
    )
    parser.add_argument(
        '--max-structures',
        type=int,
        default=None,
        help="Max structures to process"
    )
    parser.add_argument(
        '--plot-diagrams',
        action='store_true',
        help="Plot phase diagrams for each composition"
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help="Directory to cache MatterSim-relaxed MP structures (default: output-dir/mp_cache)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device for MatterSim relaxation (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect relaxed_structures.extxyz if not provided
    if args.relaxed_extxyz is None and args.structures_zip is None:
        # Try to find relaxed_structures.extxyz in same dir as stability.json
        stability_dir = Path(args.stability_json).parent
        relaxed_extxyz_path = stability_dir / "relaxed_structures.extxyz"
        if relaxed_extxyz_path.exists():
            args.relaxed_extxyz = str(relaxed_extxyz_path)
            print(f"Auto-detected: {args.relaxed_extxyz}")
        else:
            print("ERROR: Must provide either --relaxed-extxyz or --structures-zip")
            return 1
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.stability_json).parent)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API key
    import os
    mp_api_key = args.mp_api_key or os.environ.get('MP_API_KEY')
    
    if not mp_api_key:
        print("ERROR: Materials Project API key required!")
        print("  Option 1: --mp-api-key YOUR_KEY")
        print("  Option 2: export MP_API_KEY=YOUR_KEY")
        print()
        print("Get your API key at: https://next-gen.materialsproject.org/api")
        return 1
    
    print("="*70)
    print("Thermodynamic Stability Analysis (Convex Hull)")
    print("="*70)
    
    # Load structures (prefer relaxed over original)
    if args.relaxed_extxyz:
        print(f"Using RELAXED structures: {args.relaxed_extxyz}")
        print(f"Energies: {args.stability_json}")
        print(f"Output: {output_dir}")
        print("="*70)
        print()
        print("  Using relaxed structures (RECOMMENDED)")
        print("  → Hull distance computed for relaxed geometries")
        print()
        
        structures = load_relaxed_structures_and_energies(
            args.relaxed_extxyz,
            args.stability_json,
            args.max_structures
        )
        # Energy is already included in structures
        energy_map = None
        
    else:
        print(f"Using ORIGINAL structures: {args.structures_zip}")
        print(f"Energies: {args.stability_json}")
        print(f"Output: {output_dir}")
        print("="*70)
        print()
        print("  WARNING: Using original MatterGen structures, not relaxed!")
        print("  → Hull distance may be inaccurate (geometry mismatch with energy)")
        print("  → Recommended: Use --relaxed-extxyz instead")
        print()
        
        structures = load_structures_from_zip(args.structures_zip, args.max_structures)
        energy_map = load_relaxed_energies(args.stability_json)
        print(f"\nFound energies for {len(energy_map)} relaxed structures")
    
    print()
    
    # Set up cache directory for MatterSim-relaxed MP structures
    if args.cache_dir is None:
        cache_dir = output_dir / "mp_mattersim_cache"
    else:
        cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ENERGY REFERENCE: MatterSim (Option 2)")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"MP structure cache: {cache_dir}")
    print()
    print("All MP competing phases will be re-relaxed with MatterSim")
    print("to ensure consistent energy references.")
    print()
    print("This may take some time on first run, but results are cached.")
    print("="*70)
    print()
    
    # Group by CHEMICAL SYSTEM (element set), not by formula
    # This avoids redundant MP API calls for Li10B1N3, Li3B2N1, etc. (all Li-B-N system)
    chemsys_groups = {}
    for item in structures:
        # Get sorted element list (e.g., ['B', 'Li', 'N'])
        elements = sorted([str(el) for el in item['composition'].elements])
        chemsys = '-'.join(elements)  # e.g., 'B-Li-N'
        
        if chemsys not in chemsys_groups:
            chemsys_groups[chemsys] = []
        chemsys_groups[chemsys].append(item)
    
    print(f"Found {len(chemsys_groups)} unique chemical systems")
    print(f"Total structures: {sum(len(items) for items in chemsys_groups.values())}")
    print()
    
    # Analyze each chemical system
    all_results = []
    stable_structures = []
    
    import time
    
    for chemsys, items in tqdm(chemsys_groups.items(), desc="Analyzing chemical systems"):
        # Get MP competing phases ONCE per chemical system (not per formula!)
        # Re-relax them with MatterSim for consistent energy reference
        try:
            composition = items[0]['composition']
            
            # Query MP and re-relax with MatterSim (cached)
            print(f"\n{chemsys}: Fetching and relaxing MP competing phases...")
            mp_entries = get_competing_phases_from_mp_with_mattersim(
                composition, 
                mp_api_key,
                cache_dir=cache_dir,
                device=args.device
            )
            print(f"  → Using {len(mp_entries)} MatterSim-relaxed MP phases")
            
            # Skip if no MP entries found (cannot compute hull)
            if len(mp_entries) == 0:
                print(f"  → Skipping hull analysis for {chemsys} (no MP competing phases)")
                # Still record the structures but without hull data
                for item in items:
                    cif_file = item['cif_file']
                    formula = item['formula']
                    
                    if 'energy_per_atom' in item:
                        energy_per_atom = item['energy_per_atom']
                    elif energy_map and cif_file in energy_map:
                        energy_per_atom = energy_map[cif_file]
                    else:
                        continue
                    
                    result = {
                        'cif_file': cif_file,
                        'formula': formula,
                        'chemsys': chemsys,
                        'energy_per_atom': energy_per_atom,
                        'energy_above_hull': None,
                        'is_stable': None,
                        'decomposition': None,
                        'note': 'No MP competing phases found'
                    }
                    all_results.append(result)
                continue
            
            # Rate limiting: sleep between API calls to avoid 429 errors
            time.sleep(0.5)
            
            # Group items by formula within this chemical system
            formula_groups = {}
            for item in items:
                formula = item['formula']
                if formula not in formula_groups:
                    formula_groups[formula] = []
                formula_groups[formula].append(item)
            
            print(f"  → Analyzing {len(items)} structures across {len(formula_groups)} formulas")
            
            # Analyze each structure in this chemical system
            for formula, formula_items in formula_groups.items():
                print(f"    {formula}: {len(formula_items)} structure(s)")
            
            # Analyze each structure in this chemical system
            for item in items:
                cif_file = item['cif_file']
                formula = item['formula']
                
                # Get energy (either from item or from energy_map)
                if 'energy_per_atom' in item:
                    # Using relaxed structures - energy already included
                    energy_per_atom = item['energy_per_atom']
                elif energy_map and cif_file in energy_map:
                    # Using original structures - get energy from map
                    energy_per_atom = energy_map[cif_file]
                else:
                    continue
                
                structure = item['structure']
                
                try:
                    hull_data = compute_hull_distance(
                        structure, energy_per_atom, mp_entries
                    )
                    
                    result = {
                        'cif_file': cif_file,
                        'formula': formula,
                        'chemsys': chemsys,
                        'energy_per_atom': energy_per_atom,
                        'energy_above_hull': hull_data['energy_above_hull'],
                        'is_stable': hull_data['is_stable'],
                        'decomposition': hull_data['decomposition']
                    }
                    
                    all_results.append(result)
                    
                    if hull_data['is_stable']:
                        stable_structures.append(result)
                        print(f"        STABLE: {cif_file} ({formula}, E_hull = {hull_data['energy_above_hull']:.6f} eV/atom)")
                    
                    # Plot phase diagram if requested (catch plotly errors)
                    if args.plot_diagrams and hull_data['is_stable']:
                        try:
                            pd = hull_data['phase_diagram']
                            entry = ComputedEntry(
                                composition=structure.composition,
                                energy=energy_per_atom * structure.composition.num_atoms
                            )
                            
                            plot_path = output_dir / f"phase_diagram_{formula}_{cif_file.replace('.cif', '')}.png"
                            plot_phase_diagram(pd, entry, plot_path, title=f"{formula} ({chemsys})")
                        except Exception as plot_err:
                            pass
                
                except Exception as e:
                    # Record structure with error
                    result = {
                        'cif_file': cif_file,
                        'formula': formula,
                        'chemsys': chemsys,
                        'energy_per_atom': energy_per_atom,
                        'energy_above_hull': None,
                        'is_stable': None,
                        'decomposition': None,
                        'error': str(e)
                    }
                    all_results.append(result)
                    print(f"      Error analyzing {cif_file}: {e}")
        
        except Exception as e:
            print(f"  Error fetching MP data for {chemsys}: {e}")
            # Continue with next chemical system even if this one fails
    
    # Save results
    output_file = output_dir / "hull_stability.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total structures analyzed: {len(all_results)}")
    print(f"Stable structures (E_hull < 0.01 eV/atom): {len(stable_structures)}")
    print()
    
    if stable_structures:
        print("Stable structures:")
        for s in stable_structures:
            print(f"  • {s['formula']}: {s['cif_file']}")
            print(f"    E_hull = {s['energy_above_hull']:.4f} eV/atom")
    
    print()
    print(f"Results saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()

