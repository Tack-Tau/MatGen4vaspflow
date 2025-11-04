#!/usr/bin/env python
"""
Simple test script to verify MatterSim relaxation works.
Tests relaxation on a single CIF file.
"""

import sys
from pathlib import Path
from ase.io import read, write
from ase.optimize import BFGS, FIRE
import torch

print("="*70)
print("MatterSim Relaxation Test")
print("="*70)

# Check CUDA availability
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# Import MatterSim
print("\nImporting MatterSim...")
try:
    from mattersim.forcefield import MatterSimCalculator
    print(" MatterSim imported successfully")
except ImportError as e:
    print(f"  Failed to import MatterSim: {e}")
    sys.exit(1)

# Load structure
cif_file = sys.argv[1] if len(sys.argv) > 1 else "gen_0.cif"
print(f"\nLoading structure from: {cif_file}")

try:
    atoms = read(cif_file)
    print(f" Structure loaded successfully")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell volume: {atoms.get_volume():.2f} Å³")
except Exception as e:
    print(f"  Failed to load structure: {e}")
    sys.exit(1)

# Method 1: Direct calculator with BFGS optimizer
print("\n" + "="*70)
print("Method 1: MatterSimCalculator + BFGS")
print("="*70)

try:
    # Initialize calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing MatterSim calculator on {device}...")
    
    calc = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth",
        device=device
    )
    atoms.calc = calc
    print(" Calculator attached")
    
    # Get initial energy
    print("\nComputing initial energy...")
    energy_initial = atoms.get_potential_energy()
    print(f" Initial energy: {energy_initial:.4f} eV")
    print(f"  Energy per atom: {energy_initial/len(atoms):.4f} eV/atom")
    
    # Get forces
    print("\nComputing forces...")
    forces = atoms.get_forces()
    max_force = (forces**2).sum(axis=1).max()**0.5
    print(f" Forces computed")
    print(f"  Max force: {max_force:.4f} eV/Å")
    
    # Relax structure with BFGS
    print("\nRelaxing structure with BFGS optimizer...")
    print("  fmax=0.05, max_steps=100")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=100)
    
    # Final energy
    energy_final = atoms.get_potential_energy()
    print(f"\n  BFGS relaxation completed!")
    print(f"  Final energy: {energy_final:.4f} eV")
    print(f"  Energy per atom: {energy_final/len(atoms):.4f} eV/atom")
    print(f"  Energy change: {energy_final - energy_initial:.4f} eV")
    
    # Save relaxed structure
    output_file = "test_relaxed_bfgs.cif"
    write(output_file, atoms)
    print(f"  Saved to: {output_file}")
    
except Exception as e:
    print(f"  Method 1 (BFGS) failed: {e}")
    import traceback
    traceback.print_exc()

# Method 2: Using FIRE optimizer
print("\n" + "="*70)
print("Method 2: MatterSimCalculator + FIRE")
print("="*70)

try:
    # Reload original structure
    atoms = read(cif_file)
    
    # Initialize calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth",
        device=device
    )
    atoms.calc = calc
    
    # Get initial energy
    energy_initial = atoms.get_potential_energy()
    
    # Relax with FIRE (more robust for difficult cases)
    print("\nRelaxing structure with FIRE optimizer...")
    print("  fmax=0.05, max_steps=100")
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=0.05, steps=100)
    
    energy_final = atoms.get_potential_energy()
    print(f"\n  FIRE relaxation completed!")
    print(f"  Final energy: {energy_final:.4f} eV")
    print(f"  Energy per atom: {energy_final/len(atoms):.4f} eV/atom")
    print(f"  Energy change: {energy_final - energy_initial:.4f} eV")
    
    # Save relaxed structure
    output_file = "test_relaxed_fire.cif"
    write(output_file, atoms)
    print(f"  Saved to: {output_file}")
    
except Exception as e:
    print(f"  Method 2 (FIRE) failed: {e}")
    import traceback
    traceback.print_exc()

# Method 3: Batch relax (what MatterGen uses)
print("\n" + "="*70)
print("Method 3: MatterSim BatchRelaxer (used by MatterGen)")
print("="*70)

try:
    # Try to import BatchRelaxer
    try:
        from mattersim.applications.batch_relax import BatchRelaxer
    except ImportError as ie:
        print(f"  BatchRelaxer not available: {ie}")
        print("  This might be a version issue with MatterSim")
        raise
    
    # Reload original structure
    atoms = read(cif_file)
    
    print("\nInitializing BatchRelaxer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try with correct API (load_path instead of model)
    try:
        batch_relaxer = BatchRelaxer(
            load_path="MatterSim-v1.0.0-5M.pth",
            device=device,
            optimizer="BFGS",
            fmax=0.05,
            steps=100
        )
    except TypeError:
        # Fallback: check what parameters BatchRelaxer actually accepts
        import inspect
        sig = inspect.signature(BatchRelaxer.__init__)
        print(f"  BatchRelaxer parameters: {list(sig.parameters.keys())}")
        raise
    
    print(f" BatchRelaxer initialized on {device}")
    
    # Relax (batch of 1)
    print("\nRelaxing structure with BatchRelaxer...")
    relaxed_atoms_list = batch_relaxer.relax([atoms])
    
    relaxed_atoms = relaxed_atoms_list[0]
    energy_relaxed = relaxed_atoms.get_potential_energy()
    
    print(f"\n  BatchRelaxer relaxation completed!")
    print(f"  Final energy: {energy_relaxed:.4f} eV")
    print(f"  Energy per atom: {energy_relaxed/len(relaxed_atoms):.4f} eV/atom")
    
    # Save relaxed structure
    output_file = "test_relaxed_batch.cif"
    write(output_file, relaxed_atoms)
    print(f"  Saved to: {output_file}")
    
except Exception as e:
    print(f"  Method 3 (BatchRelaxer) failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  This is the method MatterGen uses for evaluation!")
    print("  If this fails, mattergen-evaluate will also fail.")

print("\n" + "="*70)
print("Test Complete")
print("="*70)
print("\nSummary:")
print("  If Methods 1 or 2 succeed: MatterSim relaxation works")
print("  If Method 3 fails: Use workaround (skip relaxation or use Method 1/2)")
print("\nCheck output files:")
print("  - test_relaxed_bfgs.cif (Method 1)")
print("  - test_relaxed_fire.cif (Method 2)")
print("  - test_relaxed_batch.cif (Method 3)")


