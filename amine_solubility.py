"""
1970 temperature / solubility pairs
118 solute/solvent experiments
104 with > 1 data points
14 with 1 data point
1 h2o
23 structural isomers
62 amines.

Unique property values:
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
import argparse
from dataclasses import dataclass, fields, asdict
import pandas as pd
from collections import namedtuple, defaultdict
import plotly
from plotly.subplots import make_subplots
from smiles_fingerprints import load_smiles

# "Solubility of <solute> in <solvent>"
# U before V!
text_columns = [
    'Solubility of:',
    'In:'
]

raw_columns = [
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

synthetic_columns = [
    'Solute SMILES',
    'Solvent SMILES',
]

CHNO = namedtuple('CHNO', ['C', 'H', 'N', 'O'])
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
TempSolubility = namedtuple('point', ['temperature', 'solubility', 'reference'])


@dataclass(kw_only=True, frozen=True)
class Compound:
    name: str
    smiles: str
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
        name = self.name
        ## Remove the part in parenthesis
        if '(' in name:
            name = name[:name.index('(')]
        return name
    
    def __repr__(self):
        return str(self)

def compound_info(compound, reference=None):
    """Create a tooltip to differentiate isomers"""
    keys = [
        'name',
        'molecular_weight_gpm',
        'complexity',
        'hydrogen_bond_donor_count',
        'rotatable_bond_count',
        'topological_polar_surface_area_angstroms',
        'undefined_atom_stereocenter_count',
        'xlogp3_aa'
    ]
    as_dict = {key: getattr(compound, key) for key in keys}
    text = chno_to_string(compound.chno) + '<br />' + '<br />'.join([f"{key}: {value}" for key, value in as_dict.items()])
    if reference:
        text += f"<br />Reference: {reference}"
    return text


def row_to_solvent(row):
    solvent = Compound(
        name=row['In:'],
        smiles=row['Solvent SMILES'],
        chno=CHNO(row['C in solvent'], row['H in solvent'], row['N in solvent'], row['O in solvent']),
        molecular_weight_gpm=row['Molecular weight solvent (g/mol)'],
        xlogp3_aa=row['XLogP3-AA solvent'],
        hydrogen_bond_donor_count=row['Hydrogen bond donor count solvent'],
        hydrogen_bond_acceptor_count=row['Hydrogen bond acceptor count solvent'],
        rotatable_bond_count=row['Rotatable bond count solvent'],
        exact_mass_da=row['Exact mass solvent (Da)'],
        monoisotopic_mass_da=row['Monoisotopic mass solvent (Da)'],
        topological_polar_surface_area_angstroms=row['Topological polar surface area solvent (Å²)'],
        heavy_atom_count=row['Heavy atom count solvent'],
        complexity=row['Complexity solvent'],
        undefined_atom_stereocenter_count=row['Undefined atom stereocenter count solvent']
    )
    return solvent

def row_to_solute(row):
    solute = Compound(
        name=row['Solubility of:'],
        smiles=row['Solute SMILES'],
        chno=CHNO(row['C in solute'], row['H in solute'], row['N in solute'], row['O in solute']),
        molecular_weight_gpm=row['Molecular weight solute (g/mol)'],
        xlogp3_aa=row['XLogP3-AA solute'],
        hydrogen_bond_donor_count=row['Hydrogen bond donor count solute'],
        hydrogen_bond_acceptor_count=row['Hydrogen bond acceptor count solute'],
        rotatable_bond_count=row['Rotatable bond count solute'],
        exact_mass_da=row['Exact mass solute (Da)'],
        monoisotopic_mass_da=row['Monoisotopic mass solute (Da)'],
        topological_polar_surface_area_angstroms=row['Topological polar surface area solute (Å²)'],
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

    bad_solute = 'Dihexylamine (N,N-Dihexylamine, N-Hexyl-1-hexanamine); C12H27N'
    df = df[df['Solubility of:'] != bad_solute]
    df = df[df['In:'] != bad_solute]

    bad_solute = 'Triisobutylamine; C12H27N'
    df = df[df['Solubility of:'] != bad_solute]
    df = df[df['In:'] != bad_solute]

    return df

def strip_single_experiment_rows(df):
    """Remove rows where there is only one instance of the 'C in solute'
    e.g. e.g., 1-Methyldodecylamine and Trioctylamine
    (If there are multiple rows with the same, it's either a different isomer
    or a second sample of the same experiment)"""
    df = df[df['C in solute'].duplicated(keep=False)]
    df = df[df['C in solvent'].duplicated(keep=False)]
    return df

def remove_nbsp(df):
    return df.replace(u'\xa0', u' ', regex=True)

def fix_commas(x):
    if isinstance(x, str) and ',' in x:
        return float(str(x).replace(',', '.'))
    return x

def load_data():
    # Read the Excel file
    path = 'data/Solubility data C4-C24.xlsx'

    numeric_cols = set(raw_columns) - set(text_columns)
    df = pd.read_excel(path, 
        converters={
            col: fix_commas for col in numeric_cols
        },
    )

    df = strip_bad_rows(df)
    df = strip_repeated_value_cols(df)
    df = strip_single_experiment_rows(df)
    df = remove_nbsp(df)
    


    filter_smoothing = True
    if filter_smoothing:
        # Step 1: Identify combinations that have SMOOTHED data
        has_smoothed = df.groupby(['Solubility of:', 'In:'])['Experiment Ref'].transform(
            lambda x: 'SMOOTHED' in x.values or 'SMOOTHED LCP' in x.values
        )

        # Step 2: Select rows that are either:
        # - SMOOTHED (if available for that combination)
        # - Experimental data (not SMOOTHED)
        df = df[(df['Experiment Ref'].isin(['SMOOTHED', 'SMOOTHED LCP'])) | (~has_smoothed)]
    else:
        df = df[~(df['Experiment Ref'].isin(['SMOOTHED', 'SMOOTHED LCP']))]


    # Rename columns with brackets to parens to avoid issues with XGBoost
    df.rename(columns=lambda col: col.replace('[', '(').replace(']', ')'), inplace=True)

    # Add SMILES codes for each compound
    smiles_map = load_smiles()
    df['Solute SMILES'] = df['Solubility of:'].map(smiles_map)
    df['Solvent SMILES'] = df['In:'].map(smiles_map)

    return df


def load_mutual_solubility_data():
    df = load_data()
    e = get_experiments(df)
    ms = get_mutual_solubility(e)

    ds = []
    for (solute, solvent), d in ms.items():
        for temperature, x, aiw in zip(d['temperature'], d['x'], d['aiw']):
            ds.append({
                **asdict(solute),
                'T (K)': temperature,
                'x': x,
                'aiw': aiw,
            })

    new_df = pd.DataFrame(ds)
    return new_df

def get_experiments(df):
    experiments = defaultdict(list)
    for i, row in df.iterrows():
        solute = row_to_solute(row)
        solvent = row_to_solvent(row)
        combination = Combination(solute, solvent)
        temperature = row['T (K)']
        solubility = row['x']
        reference = row['Experiment Ref']
        experiments[combination].append(TempSolubility(temperature, solubility, reference))

    return experiments

def get_mutual_solubility(experiments):
    d = defaultdict(lambda: {
        'temperature': [],
        'x': [],
        'aiw': [],
        'reference': [],
    })

    for combination, points in experiments.items():
        # If it's water, flip the solubility
        if combination.solute.chno == water:
            d[(combination.solvent, combination.solute)]['temperature'].extend(
                [point.temperature for point in points]
            )
            d[(combination.solvent, combination.solute)]['x'].extend(
                [1 - point.solubility for point in points]
            )
            d[(combination.solvent, combination.solute)]['aiw'].extend(
                [True for point in points]
            )
            d[(combination.solvent, combination.solute)]['reference'].extend(
                [point.reference for point in points]
            )

        else:
            d[(combination.solute, combination.solvent)]['temperature'].extend(
                [point.temperature for point in points]
            )
            d[(combination.solute, combination.solvent)]['x'].extend(
                [point.solubility for point in points]
            )
            d[(combination.solute, combination.solvent)]['aiw'].extend(
                [False for point in points]
            )
            d[(combination.solute, combination.solvent)]['reference'].extend(
                [point.reference for point in points]
            )

    return d


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
    isomers = get_all_structural_isomers(experiments)

    # Generate a color dict so each isomer has a consistent color
    colorscale = list(reversed(plotly.colors.sequential.Turbo))
    colors = {}
    offset = 0
    for compounds in isomers.values():
        for i, compound in enumerate(compounds):
            offset += 1
            if compound not in colors:
                colors[compound] = colorscale[(offset + i) % len(colorscale)]

    for combination, points in experiments.items():
        solute = combination.solute
        solvent = combination.solvent

        if solvent.chno == water:
            trace = plotly.graph_objs.Scatter(
                x=[point.temperature for point in points],
                y=[point.solubility for point in points],
                mode='markers',
                legendgroup=chno_to_string(solute.chno),
                legendgrouptitle=dict(text=chno_to_string(solute.chno)),
                marker=dict(
                    color=colors[solute]
                ),
                name=f"{solute}",
                text=[compound_info(solute, point.reference) for point in points],
                hoverinfo='text+x+y'
            )
            fig.add_trace(trace, row=1, col=1)
        elif solute.chno == water:
            trace = plotly.graph_objs.Scatter(
                x=[point.temperature for point in points],
                y=[point.solubility for point in points],
                mode='markers',
                legendgroup=chno_to_string(solvent.chno),
                legendgrouptitle=dict(text=chno_to_string(solvent.chno)),
                marker=dict(
                    color=colors[solvent]
                ),
                name=f"{solvent}",
                text=[compound_info(solvent, point.reference) for point in points],
                hoverinfo='text+x+y'
            )
            fig.add_trace(trace, row=2, col=1)

    fig.update_xaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="Solubility", row=1, col=1)
    fig.update_yaxes(title_text="Solubility", row=2, col=1)
    fig.update_layout(
        legend=dict(
            entrywidth=70,
            entrywidthmode="pixels", 
            groupclick='toggleitem',
        ),
    )

    fig.show()


def plot_solubility_vs_temperature(experiments):
    """This is a more common way to plot solubility data, where the x-axis is solubility and the y-axis is temperature.
    
    Additionally, we have flipped the solubility on the water axis."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Amines in Water", "Water in Amines (1 - X)"), shared_xaxes=True)
    isomers = get_all_structural_isomers(experiments)

    # Generate a color dict so each isomer has a consistent color
    colorscale = list(reversed(plotly.colors.sequential.Turbo))
    colors = {}
    offset = 0
    for compounds in isomers.values():
        for i, compound in enumerate(compounds):
            offset += 1
            if compound not in colors:
                colors[compound] = colorscale[(offset + i) % len(colorscale)]

    for combination, points in experiments.items():
        solute = combination.solute
        solvent = combination.solvent

        if solvent.chno == water:
            trace = plotly.graph_objs.Scatter(
                x=[point.solubility for point in points],
                y=[point.temperature for point in points],
                mode='markers',
                legendgroup=chno_to_string(solute.chno),
                legendgrouptitle=dict(text=chno_to_string(solute.chno)),
                marker=dict(
                    color=colors[solute]
                ),
                name=f"{solute}",
                text=[compound_info(solute, point.reference) for point in points],
                hoverinfo='text+x+y'
            )
            fig.add_trace(trace, row=1, col=1)
        elif solute.chno == water:
            trace = plotly.graph_objs.Scatter(
                x=[1 - point.solubility for point in points],
                y=[point.temperature for point in points],
                mode='markers',
                legendgroup=chno_to_string(solvent.chno),
                legendgrouptitle=dict(text=chno_to_string(solvent.chno)),
                marker=dict(
                    color=colors[solvent]
                ),
                name=f"{solvent}",
                text=[compound_info(solvent, point.reference) for point in points],
                hoverinfo='text+x+y'
            )
            fig.add_trace(trace, row=2, col=1)

    fig.update_xaxes(title_text="Solubility", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_layout(
        legend=dict(
            entrywidth=70,
            entrywidthmode="pixels",
            groupclick='toggleitem',
        ),
    )

    fig.show()

def double_plots():
    df = load_data()
    experiments = get_experiments(df)
    compounds = get_all_compounds(experiments)
    isomers = get_all_structural_isomers(experiments)
    differing_properties_across_isomers(isomers)
    plot_temperature_vs_solubility(experiments)
    plot_solubility_vs_temperature(experiments)

def plot_mutual_solubility():
    df = load_data()
    experiments = get_experiments(df)
    isomers = get_all_structural_isomers(experiments)

    mutual_solubility = get_mutual_solubility(experiments)

    fig = plotly.graph_objs.Figure()

    # Generate a color dict so each isomer has a consistent color
    colorscale = list(reversed(plotly.colors.sequential.Turbo))
    colors = {}
    offset = 0
    for compounds in isomers.values():
        for i, compound in enumerate(compounds):
            offset += 1
            if compound not in colors:
                colors[compound] = colorscale[(offset + i) % len(colorscale)]

    for (amine, _), d in mutual_solubility.items():
        trace = plotly.graph_objs.Scatter(
            x=d['x'],
            y=d['temperature'],
            mode='markers',
            visible=True if amine.chno == CHNO(4, 11, 1, 0) else 'legendonly',
            legendgroup=chno_to_string(amine.chno),
            legendgrouptitle=dict(text=chno_to_string(amine.chno)),
            marker=dict(
                size=10
            ),
            name=f"{amine}",
            text=[compound_info(amine, ref) for ref in d['reference']],
            hoverinfo='text+x+y'
        )
        fig.add_trace(trace)

    # Set all plots to be between 0 and 1
    fig.update_xaxes(range=[0, 1])
    fig.update_xaxes(title_text="Mole fraction amine")
    fig.update_yaxes(title_text="Temperature (K)")
    fig.update_layout(
        legend=dict(
            entrywidth=70,
            entrywidthmode="pixels",
            groupclick='toggleitem',
        ),
    )

    fig.show()



def main():
    parser = argparse.ArgumentParser(description="Filter solubility data based on smoothing and mutual solubility.")
    # Add command-line flags
    parser.add_argument("--mutual_solubility", '--ms', action="store_true", help="Show mutual solubility plots")

    args = parser.parse_args()
    if args.mutual_solubility:
        plot_mutual_solubility()
    else:
        double_plots()

if __name__ == '__main__':
    main()