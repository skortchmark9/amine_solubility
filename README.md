## Amine Solubility

Data mining to predict amine solubility

### Data Cleaning Steps
* Discarding columns which are the same across all experiments
* replace commas in numbers with periods
* some issues with "sec-Butylethylamine (C16H15N)". Chemical formula is actually C6H15N. Reports multiple molecular weights, 73.14 and 101.9. 

### Feature Engineering Thoughts:
ultimately the question we'd like to answer is:
    - given a temperature and chemical structure, what is the solubility in water (x)

Features (10):
* CHNO
* xlogp3_aa
* hydrogen_bond_donor_count
* rotatable_bond_count
* topological_polar_surface_area_angstroms
* complexity
* undefined_atom_stereocenter_count

Discarding columns which are the same across all isomers, e.g. molecular mass.

### Questions:
* how to deal with water as solvent?
    * they do seem to have similar trends in a lot of cases
* how to deal with different number of data points for different experiments
    * ellipses? (similar to hurricanes paper)
    * filter out experiments with only 1 data point?
