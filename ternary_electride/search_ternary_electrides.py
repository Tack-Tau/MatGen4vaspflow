#!/usr/bin/env python3
"""
Search for potential ternary electride compositions based on valence electron counting.
Generates a list of compositions to pass to MatterGen for structure generation.
"""

import json
from math import gcd
from functools import reduce
from typing import List, Dict, Tuple


# Valence electrons dictionary for elements (most common oxidation states)
VALENCE_ELECTRONS = {
    # Group I
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    # Group II
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    # Group III (including lanthanides, excluding actinides)
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'Sc': 3, 'Y': 3,
    # Lanthanides
    'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3,
    'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3,
    # Group IV
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    # Group V
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,  # Common negative valence
    # Group VI
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'Po': 2,  # Common negative valence
    # Group VII
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'At': 1,  # Common negative valence
}

# Define element groups for searching
GROUP_I_II_III = ['Li', 'Na', 'K', 'Rb', 'Cs',     # Group I
                  'Be', 'Mg', 'Ca', 'Sr', 'Ba',    # Group II
                  'Sc', 'Y'                        # Group III transition
                  'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']  # Lanthanides

GROUP_III_IV = ['B', 'Al', 'Ga', 'In',             # Group III
                'C', 'Si', 'Ge', 'Sn', 'Pb']       # Group IV (Z <= 82)

GROUP_V_VI_VII = ['N', 'P', 'As', 'Sb',            # Group V (Z <= 51)
                  'O', 'S', 'Se', 'Te',            # Group VI (Z <= 52)
                  'F', 'Cl', 'Br', 'I']            # Group VII (Z <= 53)


def gcd_multiple(numbers: List[int]) -> int:
    """Calculate GCD of multiple numbers."""
    return reduce(gcd, numbers)


def count_excess_electrons(A: str, l: int, B: str, m: int, C: str, n: int) -> float:
    """
    Count excess valence electrons in composition A_l B_m C_n.
    
    Positive excess = potential electride
    Formula: n_excess = valence(A)*l + valence(B)*m - valence(C)*n
    """
    val_A = VALENCE_ELECTRONS[A]
    val_B = VALENCE_ELECTRONS[B]
    val_C = VALENCE_ELECTRONS[C]
    
    # For electronegative elements (groups V, VI, VII), use as electron acceptors
    excess = val_A * l + val_B * m - abs(val_C) * n
    
    return excess


def search_ternary_electrides(
    group_A: List[str] = GROUP_I_II_III,
    group_B: List[str] = GROUP_III_IV,
    group_C: List[str] = GROUP_V_VI_VII,
    max_atoms: int = 20,
    excess_electron_range: Tuple[float, float] = (0.1, 4.0),
    max_compositions: int = -1
) -> List[Dict]:
    """
    Search for potential ternary electride compositions.
    
    Args:
        group_A: List of elements for position A (metals)
        group_B: List of elements for position B (semi-metals/metalloids)
        group_C: List of elements for position C (non-metals)
        max_atoms: Maximum number of atoms in unit cell
        excess_electron_range: (min, max) excess electrons for electride
        max_compositions: Maximum number of compositions to generate (-1 for all)
        
    Returns:
        List of valid compositions as dictionaries
    """
    valid_compositions = []
    n_count = 0
    
    print("Searching for potential ternary electrides...")
    print(f"Excess electron range: {excess_electron_range[0]} to {excess_electron_range[1]}")
    print(f"Max atoms per composition: {max_atoms}")
    print("="*70)
    
    for A in group_A:
        for B in group_B:
            for C in group_C:
                # Try different stoichiometries
                for l in range(1, max_atoms):
                    for m in range(1, max_atoms):
                        for n in range(1, max_atoms):
                            # Check total atoms
                            if l + m + n > max_atoms:
                                continue
                            
                            # Reduce to smallest integer ratio
                            g = gcd_multiple([l, m, n])
                            l_p, m_p, n_p = l // g, m // g, n // g
                            
                            # Skip if not reduced form
                            if (l_p, m_p, n_p) != (l, m, n):
                                continue
                            
                            # Calculate excess electrons
                            excess = count_excess_electrons(A, l_p, B, m_p, C, n_p)
                            
                            # Check if in valid range for electride
                            if excess_electron_range[0] < excess <= excess_electron_range[1]:
                                composition = {
                                    A: l_p,
                                    B: m_p,
                                    C: n_p
                                }
                                
                                formula = f"{A}{l_p}{B}{m_p}{C}{n_p}"
                                
                                valid_compositions.append({
                                    'formula': formula,
                                    'composition': composition,
                                    'excess_electrons': excess,
                                    'total_atoms': l_p + m_p + n_p,
                                    'elements': [A, B, C]
                                })
                                
                                n_count += 1
                                
                                if n_count % 100 == 0:
                                    print(f"Found {n_count} compositions... (Latest: {formula})")
                                
                                if max_compositions > 0 and n_count >= max_compositions:
                                    print(f"\nReached maximum of {max_compositions} compositions.")
                                    return valid_compositions
    
    print(f"\nTotal valid compositions found: {len(valid_compositions)}")
    return valid_compositions


def save_compositions(compositions: List[Dict], output_file: str = "ternary_electride_compositions.json"):
    """Save compositions to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(compositions, f, indent=2)
    print(f"\nCompositions saved to: {output_file}")
    
    # Also save a simple text file with just formulas for easy viewing
    txt_file = output_file.replace('.json', '.txt')
    with open(txt_file, 'w') as f:
        for comp in compositions:
            f.write(f"{comp['formula']}\n")
    print(f"Formulas list saved to: {txt_file}")


def print_statistics(compositions: List[Dict]):
    """Print statistics about found compositions."""
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    # Count by total atoms
    from collections import Counter
    atom_counts = Counter(c['total_atoms'] for c in compositions)
    print("\nDistribution by total atoms:")
    for atoms in sorted(atom_counts.keys()):
        print(f"  {atoms:2d} atoms: {atom_counts[atoms]:4d} compositions")
    
    # Count by excess electrons
    excess_counts = Counter(round(c['excess_electrons'], 1) for c in compositions)
    print("\nDistribution by excess electrons:")
    for excess in sorted(excess_counts.keys()):
        print(f"  {excess:4.1f} e⁻: {excess_counts[excess]:4d} compositions")
    
    # Show some examples
    print("\nExample compositions (sorted by excess electrons):")
    sorted_comps = sorted(compositions, key=lambda x: x['excess_electrons'], reverse=True)
    for comp in sorted_comps[:10]:
        print(f"  {comp['formula']:15s} | {comp['excess_electrons']:4.1f} e⁻ | {comp['total_atoms']:2d} atoms")


def main():
    """Main execution."""
    print("="*70)
    print("TERNARY ELECTRIDE COMPOSITION SEARCH")
    print("="*70)
    
    # Search for compositions
    compositions = search_ternary_electrides(
        max_atoms=20,              # Maximum atoms per unit cell
        excess_electron_range=(0.1, 4.0),  # 0.1-4 excess electrons
        max_compositions=-1         # Search all compositions (-1 means no limit)
    )
    
    # Print statistics
    print_statistics(compositions)
    
    # Save results
    save_compositions(compositions, "ternary_electride_compositions.json")
    
    print("\n" + "="*70)
    print(f" Search completed! Found {len(compositions)} potential electride compositions.")
    print("="*70)
    print("\nNext steps:")
    print("1. Review ternary_electride_compositions.json")
    print("2. Transfer to cluster: scp ternary_electride_compositions.json HPC_HOST:~/SOFT/mattergen_test/")
    print("3. Run generation: sbatch generate_ternary_electrides.sh")
    print("="*70)


if __name__ == "__main__":
    main()

