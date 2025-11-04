import pandas as pd
import os

# cross refence with a file for materials_id and spacegroup
# This is a boolean, 1 if in the reference file and 0 if not
# Use both materials_id and spacegroup for the cross-reference

def add_is_electride_field(curated_file="curated_mp_data.csv", electride_file="electrides.txt", output_file="curated_mp_data_electrides.csv"):
    curated_df = pd.read_csv(curated_file, index_col=0)
    electride_df = pd.read_csv(electride_file, sep=r'\s+', header=None, names=["material_id"])#, "pretty_formula"])
    merged_df = curated_df.merge(electride_df, on=["material_id"], how="left", indicator=True)
    curated_df["is_electride"] = merged_df["_merge"].apply(lambda x: 1 if x == "both" else 0)
    #curated_df["is_electride"] = merged_df["_merge"].notnull().astype(int)
    curated_df.to_csv(output_file, index=True, lineterminator="\n")
    # avoid ^M in the output file
    with open(output_file, 'rb+') as f:
        content = f.read()
        f.seek(0)
        f.write(content.replace(b'\r\n', b'\n'))
        f.truncate()

    # also return the material_ids that were matches
    matched_material_ids = curated_df[curated_df["is_electride"] == 1]["material_id"].tolist()
    # print the number of matches
    print(f"Number of matched material_ids: {len(matched_material_ids)} in {curated_file}")
    # write to txt file - extract just the base filename without directory
    base_filename = os.path.basename(curated_file).split('.')[0]
    with open(f"matched_{base_filename}_ids.txt", "w") as f:
        f.write("\n".join(map(str, matched_material_ids)))

# another function that adds metal-nonmetal field
# use the cif field of the curated_file to determine if a structure is metal or non-metal
# can use the _chemical_formula_structural or _chemical_formula_sum or first line in the cif field 
'''
data_LiMnIr2
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.16442336
_cell_length_b   4.16442336
_cell_length_c   4.16442336
_cell_angle_alpha   60.00000000
_cell_angle_beta   60.00000000
_cell_angle_gamma   60.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   LiMnIr2
_chemical_formula_sum   'Li1 Mn1 Ir2'
_cell_volume   51.06809118
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Li  Li0  1  0.50000000  0.50000000  0.50000000  1
  Mn  Mn1  1  0.00000000  0.00000000  0.00000000  1
  Ir  Ir2  1  0.25000000  0.25000000  0.25000000  1
  Ir  Ir3  1  0.75000000  0.75000000  0.75000000  1
'''
def add_metal_nonmetal_field(curated_file="curated_mp_data.csv", output_file="metal_nonmetal.csv"):
    non_metals = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Si', 'Br', 'I']
    # for metals, get all Lanthanides, Actinides and Groups 1-2 only
    all_La = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    all_Ac = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    group_1_2 = ['Li', 'Be', 'Na', 'Mg', 'K', 'Ca', 'Rb', 'Sr', 'Cs', 'Ba', 'Sc', 'Y']
    metals = all_La + all_Ac + group_1_2
    # Now, data of interest such that the cif has both metals and non-metals
    # and one of the metals is from the metals list above and one of the non-metals is from the non_metals list above
    # Example LaMnSi or MnSi but not CaMn or LaMn
    # Classify this as 1 or 0
    curated_df = pd.read_csv(curated_file, index_col=0)
    def classify_metal_nonmetal(cif_str):
        # Clean the string
        cif_str = cif_str.strip().strip('"')
        lines = cif_str.splitlines()
        if lines[0].startswith("#"):
            lines = lines[1:]
        cif_clean = "\n".join(lines)
        # Get the first line that starts with _chemical_formula_structural or _chemical_formula_sum
        formula_line = None
        for line in lines:
            if line.startswith("_chemical_formula_structural") or line.startswith("_chemical_formula_sum"):
                formula_line = line
                break
        if formula_line is None:
            return 0
        # Get the formula from the line
        formula = formula_line.split()[-1].strip("'\"")
        # Get the elements in the formula
        elements_in_formula = []
        current_element = ""
        for char in formula:
            if char.isupper():
                if current_element:
                    elements_in_formula.append(current_element)
                current_element = char
            elif char.islower():
                current_element += char
            elif char.isdigit():
                continue
            else:
                if current_element:
                    elements_in_formula.append(current_element)
                    current_element = ""
        if current_element:
            elements_in_formula.append(current_element)
        elements_in_formula = list(set(elements_in_formula)) # unique elements only
        has_metal = any(el in metals for el in elements_in_formula)
        has_nonmetal = any(el in non_metals for el in elements_in_formula)
        if has_metal and has_nonmetal:
            return 1
        else:
            return 0
    curated_df["metal_nonmetal"] = curated_df["cif"].apply(classify_metal_nonmetal)
    curated_df.to_csv(output_file, index=True, lineterminator="\n")
    # avoid ^M in the output file
    with open(output_file, 'rb+') as f:
        content = f.read()
        f.seek(0)
        f.write(content.replace(b'\r\n', b'\n'))
        f.truncate()
    
'''
curated_file = "curated_mp_data.csv"
output_file = f"{curated_file.split('.')[0]}_electrides.csv"
add_is_electride_field(curated_file=curated_file, electride_file="is_electride_file.txt", output_file=output_file)
'''
'''
# do it for all curated files
curated_files = ["val.csv", "train.csv", "test.csv"]
for curated_file in curated_files:
    output_file = f"{curated_file.split('.')[0]}_electrides.csv"
    add_is_electride_field(curated_file=curated_file, electride_file="is_electride_file.txt", output_file=output_file)

# concatenate  the matched_material_ids.txt files
with open("matched_material_ids.txt", "w") as outfile:
    for curated_file in curated_files:
        with open(f"matched_{curated_file.split('.')[0]}_ids.txt") as infile:
            outfile.write(infile.read())
            # add a newline after each file's content
            outfile.write("\n")

# remove duplicates
with open("matched_material_ids.txt") as infile, open("matched_material_ids_no_duplicates.txt", "w") as outfile:
    seen = set()
    for line in infile:
        if line not in seen:
            seen.add(line)
            outfile.write(line)

# see what is in is_electride_file.txt that is not in matched_material_ids_no_duplicates.txt
# and write to new file, call not_matched_ids.txt
with open("is_electride_file.txt") as infile, open("not_matched_ids.txt", "w") as outfile:
    seen = set()
    with open("matched_material_ids_no_duplicates.txt") as matched_file:
        for line in matched_file:
            seen.add(line)
    for line in infile:
        if line not in seen:
            outfile.write(line)
'''
# STEP 1: Extract material IDs from electrides-reduced.csv

electride_df = pd.read_csv("electrides-reduced.csv")
# Extract the Material id column (has spaces in the name, so need to handle that)
material_ids = electride_df[' Material id  '].str.strip().tolist()
# Write to a simple text file
with open("electride_material_ids.txt", "w") as f:
    for mat_id in material_ids:
        f.write(f"{mat_id}\n")
print(f"Extracted {len(material_ids)} material IDs to electride_material_ids.txt")

# STEP 2: Add is_electride field to mp_20 dataset files
curated_files = ["mp_20/train.csv", "mp_20/val.csv", "mp_20/test.csv"]
for curated_file in curated_files:
    # Keep the files in the mp_20 directory
    output_file = curated_file  # Overwrite the original file
    add_is_electride_field(curated_file=curated_file, 
                          electride_file="electride_material_ids.txt", 
                          output_file=output_file)
    print(f"Added is_electride field to {curated_file}")


