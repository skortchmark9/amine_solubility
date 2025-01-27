from dataclasses import dataclass
import pandas as pd
from collections import namedtuple, defaultdict

columns = [
    'Solubility of:',
    'In:',
    'T [K]',
    'x',

    'C in solute',
    'H in solute',
    'N in solute',
    'O in solute',
    'Molecular weight solute [g/mol]',
    'XLogP3-AA solute',
    'Hydrogen bond donor count solute',
    'Hydrogen bond acceptor count solute',
    'Rotatable bond count solute',
    'Exact mass solute [Da]',
    'Monoisotopic mass solute [Da]',
    'Topological polar surface area solute [Å²]',
    'Heavy atom count solute',
    'Complexity solute',
    'Undefined atom stereocenter count solute',


    'C in solvent',
    'H in solvent',
    'N in solvent',
    'O in solvent',
    'Molecular weight solvent [g/mol]',
    'XLogP3-AA solvent',
    'Hydrogen bond donor count solvent',
    'Hydrogen bond acceptor count solvent',
    'Rotatable bond count solvent',
    'Exact mass solvent [Da]',
    'Monoisotopic mass solvent [Da]',
    'Topological polar surface area solvent [Å²]',
    'Heavy atom count solvent',
    'Complexity solvent',
    'Undefined atom stereocenter count solvent'
]

CHNO = namedtuple('CHNO', ['C', 'H', 'N', 'O'])

def chno_to_string(chno):
    parts = []
    if chno.C > 0:
        parts.append(f"C{chno.C}")
    if chno.H > 0:
        parts.append(f"H{chno.H}")
    if chno.N > 0:
        parts.append(f"N{chno.N}")
    if chno.O > 0:
        parts.append(f"O{chno.O}")
    return ''.join(parts)

Combination = namedtuple('Combination', ['solute', 'solvent'])
TempSolubility = namedtuple('point', ['temperature', 'solubility'])


@dataclass(kw_only=True, frozen=True)
class Compound:
    chno: CHNO
    molecular_weight_gpm: float
    xlogp3_aa: float
    hydrogen_bond_donor_count: float
    hydrogen_bond_acceptor_count: float
    rotatable_bond_count: float
    exact_mass_da: float
    monoisotopic_mass_da: float
    topological_polar_surface_area_angstroms: float
    heavy_atom_count: float
    complexity: float
    undefined_atom_stereocenter_count: float

    def __str__(self):
        """Format the compound as a string based on chno"""
        return f"Compound({chno_to_string(self.chno)})"
    
    def __repr__(self):
        return str(self)

def row_to_solvent(row):
    solvent = Compound(
        chno=CHNO(row['C in solvent'], row['H in solvent'], row['N in solvent'], row['O in solvent']),
        molecular_weight_gpm=row['Molecular weight solvent [g/mol]'],
        xlogp3_aa=row['XLogP3-AA solvent'],
        hydrogen_bond_donor_count=row['Hydrogen bond donor count solvent'],
        hydrogen_bond_acceptor_count=row['Hydrogen bond acceptor count solvent'],
        rotatable_bond_count=row['Rotatable bond count solvent'],
        exact_mass_da=row['Exact mass solvent [Da]'],
        monoisotopic_mass_da=row['Monoisotopic mass solvent [Da]'],
        topological_polar_surface_area_angstroms=row['Topological polar surface area solvent [Å²]'],
        heavy_atom_count=row['Heavy atom count solvent'],
        complexity=row['Complexity solvent'],
        undefined_atom_stereocenter_count=row['Undefined atom stereocenter count solvent']
    )
    return solvent

def row_to_solute(row):
    solute = Compound(
        chno=CHNO(row['C in solute'], row['H in solute'], row['N in solute'], row['O in solute']),
        molecular_weight_gpm=row['Molecular weight solute [g/mol]'],
        xlogp3_aa=row['XLogP3-AA solute'],
        hydrogen_bond_donor_count=row['Hydrogen bond donor count solute'],
        hydrogen_bond_acceptor_count=row['Hydrogen bond acceptor count solute'],
        rotatable_bond_count=row['Rotatable bond count solute'],
        exact_mass_da=row['Exact mass solute [Da]'],
        monoisotopic_mass_da=row['Monoisotopic mass solute [Da]'],
        topological_polar_surface_area_angstroms=row['Topological polar surface area solute [Å²]'],
        heavy_atom_count=row['Heavy atom count solute'],
        complexity=row['Complexity solute'],
        undefined_atom_stereocenter_count=row['Undefined atom stereocenter count solute']
    )
    return solute


def strip_repeated_value_cols(df):
    """Remove all the columns where all the row values are the same and return a new dataframe.

    Omits columns for unit count, stereocenter, formal charge, isotope atom count, undefined bond stereocenter 
    
    """ 
    return df.loc[:, (df != df.iloc[0]).any()]

def fix_commas(x):
    if isinstance(x, str) and ',' in x:
        return float(str(x).replace(',', '.'))
    return x

def load_data():
    # Read the Excel file

    numeric_cols = set(columns) - set(['Solubility of:', 'In:'])
    df = pd.read_excel("data/Solubility data C4-C24.xlsx", converters={
        col: fix_commas for col in numeric_cols
    })

    df = strip_repeated_value_cols(df)

    return df

def get_experiments(df):
    experiments = defaultdict(list)
    for i, row in df.iterrows():
        solute = row_to_solute(row)
        solvent = row_to_solvent(row)
        combination = Combination(solute, solvent)
        temperature = row['T [K]']
        solubility = row['x']
        experiments[combination].append(TempSolubility(temperature, solubility))

    return experiments