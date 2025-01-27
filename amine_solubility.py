"""
1970 temperature / solubility pairs
118 solute/solvent experiments
104 with > 1 data points
14 with 1 data point
1 h2o
23 structural isomers
62 amines. Unique property values:
  = shared across isomers
  * differing across isomers

  = C: 11
  = H: 13
  = N: 2
  = O: 1
  = molecular_weight_gpm: 20
  = exact_mass_da: 20
  = monoisotopic_mass_da: 20
  = heavy_atom_count: 11
  = hydrogen_bond_acceptor_count: 2
  * xlogp3_aa: 29
  * hydrogen_bond_donor_count: 2
  * rotatable_bond_count: 12
  * topological_polar_surface_area_angstroms: 5
  * complexity: 55
  * undefined_atom_stereocenter_count: 3

Samples to features:
    1986 : 120
"""
from dataclasses import dataclass, fields
import pandas as pd
from collections import namedtuple, defaultdict
import plotly
from plotly.subplots import make_subplots

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

_CHNO = namedtuple('_CHNO', ['C', 'H', 'N', 'O'])
class CHNO(_CHNO):
    def __str__(self):
        return chno_to_string(self)
    
    def __repr__(self):
        return str(self)


water = CHNO(0, 2, 0, 1)

def chno_to_string(chno):
    parts = []
    if chno.C > 0:
        parts.append(f"C{chno.C}" if chno.C > 1 else "C")
    if chno.H > 0:
        parts.append(f"H{chno.H}" if chno.H > 1 else "H")
    if chno.N > 0:
        parts.append(f"N{chno.N}" if chno.N > 1 else "N")
    if chno.O > 0:
        parts.append(f"O{chno.O}" if chno.O > 1 else "O")
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
        return f"{self.chno}"
    
    def __repr__(self):
        return str(self)

def compound_info(compound):
    """Create a tooltip to differentiate isomers"""
    keys = [
        'complexity',
        'hydrogen_bond_donor_count',
        'rotatable_bond_count',
        'topological_polar_surface_area_angstroms',
        'undefined_atom_stereocenter_count',
        'xlogp3_aa'
    ]
    as_dict = {key: getattr(compound, key) for key in keys}
    text = str(compound.chno) + '<br />' + '<br />'.join([f"{key}: {value}" for key, value in as_dict.items()])
    return text


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

def strip_bad_rows(df):
    bad_solute = 'sec-Butylethylamine (C16H15N)'
    df = df[df['Solubility of:'] != bad_solute]
    df = df[df['In:'] != bad_solute]

    return df

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
    df = strip_bad_rows(df)

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

def get_all_compounds(experiments):
    compounds = set()
    for combination in experiments.keys():
        compounds.add(combination.solute)
        compounds.add(combination.solvent)
    return compounds

def get_all_structural_isomers(experiments):
    compounds = get_all_compounds(experiments)
    isomers = defaultdict(list)
    for compound in compounds:
        isomers[compound.chno].append(compound)

    return isomers

def differing_properties_across_isomers(isomers):
    """For each isomer, print out fields which differ"""
    differing_properties = set()
    for chno, compounds in isomers.items():
        print(f"CHNO: {chno}")
        for field in fields(Compound):
            values = set(getattr(compound, field.name) for compound in compounds)
            if len(values) > 1:
                print(f"\tProperty {field.name} differs: {values}")
                differing_properties.add(field.name)

    return differing_properties


def plot_temperature_vs_solubility(experiments):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Amines in Water", "Water in Amines"), shared_xaxes=True)

    for combination, points in experiments.items():
        solute = combination.solute
        solvent = combination.solvent

        if solvent.chno == water:
            trace = plotly.graph_objs.Scatter(
                x=[point.temperature for point in points],
                y=[point.solubility for point in points],
                mode='markers',
                name=f"{solute}",
                text=[compound_info(solute) for _ in points],
                hoverinfo='text+x+y'
            )
            fig.add_trace(trace, row=1, col=1)
        elif solute.chno == water:
            trace = plotly.graph_objs.Scatter(
                x=[point.temperature for point in points],
                y=[point.solubility for point in points],
                mode='markers',
                name=f"{solvent}",
                text=[compound_info(solvent) for _ in points],
                hoverinfo='text+x+y'
            )
            fig.add_trace(trace, row=2, col=1)

    fig.update_xaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="Solubility", row=1, col=1)
    fig.update_yaxes(title_text="Solubility", row=2, col=1)

    fig.show()
