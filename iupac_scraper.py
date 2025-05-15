import math
import re
import pdfplumber
import pandas as pd
import numpy as np
from smiles_fingerprints import load_smiles, smiles_to_chno, chno_to_string, CHNO
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.colors as pc


sources = [
    "papers/c4-c6 amines.pdf",
    "papers/c7-c24 amines.pdf",
    "papers/non-aliphatic amines.pdf",
]

def load():
    return pdfplumber.open(sources[0])
    

def partition(x, cond):
    return [x for x in x if cond(x)], [x for x in x if not cond(x)]

def extract_tables_with_preceding_text(pdf):
    tables_with_text = []

    for i, page in enumerate(pdf.pages):
        found_tables = page.find_tables()
        lines = [
            line for line in page.extract_text_lines() if not
            any(table.bbox[1] < line["top"] < table.bbox[3] for table in found_tables)
        ]
        all_objects = lines + found_tables
        all_objects.sort(key=lambda obj: obj["top"] if isinstance(obj, dict) else obj.bbox[1])
        topmost_object = all_objects[0] if all_objects else None
        bottommost_object = all_objects[-1] if all_objects else None

        for table in found_tables:
            table_bbox = table.bbox
            lines_above, rest = partition(lines, lambda line: line["bottom"] < table_bbox[1])
            last_table = table == found_tables[-1]

            extracted_table = table.extract(y_tolerance=4)
            cleaned_table = [row for row in extracted_table if any(cell.strip() for cell in row if cell)]

            tables_with_text.append({
                'table': cleaned_table,
                'page': i + 1,
                'lines_above': [l['text'] for l in lines_above],
                'lines_below': [l['text'] for l in rest] if last_table else [],
                'is_top': table == topmost_object,
                'is_bottom': table == bottommost_object,
            })
            lines = rest

    out = []
    prev_table = None
    for table in tables_with_text:
        preceding_lines = ['Unknown']
        # Simple case: there is text above it, so use that to name it
        if table['lines_above']:
            preceding_lines = table['lines_above'][-2:]
        # No lines above, and none below the previous one - merge them.
        # elif prev_table and not prev_table['lines_below']:
        elif prev_table and table['is_top'] and prev_table['is_bottom'] and prev_table['page'] == (table['page'] - 1):
            prev_table['table'].extend(table['table'])
            prev_table['lines_below'] = table['lines_below']
            prev_table['is_bottom'] = table['is_bottom']
            continue
        elif prev_table and prev_table['lines_below']:
            preceding_lines = prev_table['lines_below'][:2]

        out.append({
            'preceding_text': '\n'.join(preceding_lines),
            'table': table['table'],
            'page': table['page'],
        })
        prev_table = table
        continue

    return out


def get_multi_value_rows(row):
    cells_with_commas = [cell for cell in row if ',' in (cell or '')]
    if not cells_with_commas:
        return []
    
    if len(cells_with_commas) > 1:
        raise Exception("Multiple multi-value cells")
    
    cell_with_comma = cells_with_commas[0]
    i = row.index(cell_with_comma)

    # If the other cells are non-numeric, this is probably keys
    # and thus not a multi-value cell
    for cell in row:
        if cell is None:
            return []
        if cell != cell_with_comma and not likely_number(cell):
            return []


    if cells_with_commas[0] == '':
        return []

    cell_with_comma = cell_with_comma.replace('\n', ' ')
    values = cell_with_comma.split(',')
    new_rows = []
    for value in values:
        new_row = row.copy()
        new_row[i] = value.strip()
        new_rows.append(new_row)

    return new_rows


def parse_cell(cell):
    source = cell
    superscript = None
    if cell is None:
        return { 'content': '', 'tags': [], 'superscript': superscript, 'source': source, 'value': None }

    cell = cell.replace('−', '-')
    # Separate out parenthetical tags from cell values
    tags = re.findall(r"\(([^)]+)\)", cell)
    if tags:
        cell = re.sub(r"\([^)]+\)", "", cell).strip()
        # split tags which contain ;
        tags = [tag.split(';') for tag in tags]
        tags = [item.strip().replace('\n', ' ') for sublist in tags for item in sublist]
    else:
        tags = []

    content = cell.strip()
    content = content.replace('−', '-')

    return {
        'content': content,
        'tags': tags,
        'superscript': superscript,
        'source': source,
        'value': None,
    }

def likely_number(s):
    # Check that it contains only digits, decimal point, x, and ±
    return all(c.isdigit() or c in "\n ,.×±-−" for c in s)

def likely_key(s):
    return any([
        'compiler' in s,
        'Solubility' in s,
        'w1' in s,
        'w2' in s,
        'T/' in s,
        'T/K' in s,
        't/' in s,
        'x1' in s,
        'x2' in s,
        'Smoothed' in s,
        'Experimental values' in s,
        'T' in s,
    ])


def clean_and_split_table(table):
    out = []
    header = table["preceding_text"]

    if (
        header.lower().startswith("experimental values") or 
        re.match("^Table \\d+\\.", header, re.IGNORECASE)
        or header.lower().startswith("solubility of")
    ):
        if '+' in header:
            return out

        nsplit = header.split('\n')
        if len(nsplit) > 1:
            name = nsplit[1]
        else:
            name = nsplit[0]
        keys = table['table'][0]
        if keys[0] in ('Author(s)', 'Components'):
            return out

        input_rows = table['table'][1:]
        output_rows = []

        def finish():
            out.append({
                'name': name,
                'page': table['page'],
                'keys': keys,
                'rows': output_rows
            })

        while input_rows:
            row = input_rows.pop(0)

            multi_value_rows = get_multi_value_rows(row)
            if multi_value_rows:
                input_rows = multi_value_rows + input_rows
                continue


            parsed_row = [parse_cell(cell) for cell in row]

            row_is_keys = all([likely_key(cell['content']) for cell in parsed_row if cell['source'] is not None])
            if not row_is_keys:
                if len(row) != len(keys):
                    # print(row, keys)
                    # Parsing messed up somehow
                    # print("Stopping parsing after row arity mismatch")
                    # print('Keys:')
                    # print(keys)
                    # print('Row:')
                    # print(row)
                    # print("Whole Table")
                    # print(table)
                    break

                output_rows.append(parsed_row)
            else:
                # Rows can flip in the middle of the table, so create a new one in that case
                if parsed_row[0]['content'].lower().startswith('solubility of'):
                    finish()
                    name = row[0]
                    output_rows = []
                    keys = input_rows.pop(0)
                else:
                    finish()
                    keys = row
                    output_rows = []


        if output_rows:
            finish()

    return out

def print_table(table):
    print(table['name'], 'Page', table['page'])
    print(table['keys'])
    print(len(table['rows']), 'rows')
    if table['rows']:
        print('First:')
        print(table['rows'][0])
        print('Last:')
        print(table['rows'][-1])

def organize_tables(pdf):
    tables = extract_tables_with_preceding_text(pdf)

    cleaned = []
    for table in tables:
        cleaned.extend(clean_and_split_table(table))

    return cleaned

def parse_scientific_notation(s):
    """Parses a string in the format '9.24 × 10−3' and returns a float."""
    match = re.match(r"([\d\.]+)\s*×\s*10([−\-]?\d+)", s)
    if not match:
        return math.nan
        raise ValueError(f"Invalid scientific notation format: {s}")
    
    base, exponent = match.groups()
    exponent = exponent.replace('−', '-')  # Handle minus signs
    return float(base) * (10 ** int(exponent))

def is_number_with_ending_superscript(s):
    """Checks if a string is a number with a superscript at the end."""
    return re.match(r"[−\d\.]+[a-z,]{1,3}", s)

def get_number_and_superscript(s):
    """Checks if a string is a number with a superscript at the end."""
    return re.match(r"([−\d\.]+)([a-z,]{1,3})", s)

def is_exponent(s):
     return '×' in s

def transform_row(row):
    # Handle exponents and references
    update = []
    for cell in row:
        cell = cell.copy()
        cell['content'] = cell['content'].replace('−', '-').strip()
        if is_number_with_ending_superscript(cell['content']):
            number, superscript = get_number_and_superscript(cell['content']).groups()
            try:
                cell['value'] = float(number)
                cell['superscript'] = superscript
            except ValueError:
                raise AmbiguousException(f'Invalid number with superscript {cell["content"]}')
        elif is_exponent(cell['content']):
            cell['value'] = parse_scientific_notation(cell['content'])
        elif '±' in cell['content']:
            cell['value'] = handle_plus_minus(cell['content'])
        elif cell['content']:
            if '>' in cell['content']:
                raise AmbiguousException("Greater than sign in cell")

            # TODO: fix this - something weird going on with superscripts
            try:
                cell['value'] = float(cell['content'])
            except ValueError:
                raise AmbiguousException(f'Invalid number {cell["content"]}')

        update.append(cell)
    return update

class AmbiguousException(Exception):
    pass

def handle_plus_minus(str):
    out = str.replace('(', '').replace(')', '')
    return float(out.split('±')[0])

def parse_tables(pdf):
    tables = organize_tables(pdf)
    error_rows = 0
    success_rows = 0
    for table in tables:
        new_rows = []
        for i, raw_row in enumerate(table['rows']):
            try:
                transformed = transform_row(raw_row)
                new_rows.append(transformed)
                success_rows += 1
            except AmbiguousException as e:
                error_rows += 1
                print(e)
            except Exception as e:
                print("Fatal error parsing row", i, "of table", table['name'])
                print_table(table)
                raise e

        table['rows'] = new_rows

    print(f"Parsing errors {error_rows} out of {success_rows + error_rows} rows")
            
    return tables

def get_compound_from_name(name):
    match = re.match(r"Experimental values for solubility of (.+?) \(\d\) in (.+) \(\d\)", name)
    if match:
        compound1 = match.group(1)
        compound2 = match.group(2)
        solute = compound1.strip()
        solvent = compound2.strip()
        return solute, solvent

    match = re.match(r"Solubility of (.+?) in (.+)", name)
    if match:
        compound1 = match.group(1)
        compound2 = match.group(2)
        solute = compound1.strip()
        solvent = compound2.strip()
        return solute, solvent

    return None, None

def c_to_k(c):
    """Convert Celsius to Kelvin."""
    return c + 273.15


def to_df(table):
    columns = [
        'T',
        'x',
        'T (smoothed)',
        'x (smoothed)',
        'Confidence',
        'Reference',
        'Estimated LCP',
        'Estimated UCP',
    ]

    row_datas = []
    keys_lower = [k.lower() for k in table['keys']]
    for row in table['rows']:
        row_data = {k: None for k in columns}
        for (cell, key) in zip(row, keys_lower):
            if key == 't/k':
                row_data['T'] = cell['value']
            elif key == 't/°c':
                row_data['T'] = c_to_k(cell['value'])
            elif 't/k' in key and 'smoothed' in key:
                row_data['T (smoothed)'] = cell['value']
            elif ('x1' in key or 'x2' in key):
                if 'smoothed' in key:
                    row_data['x (smoothed)'] = cell['value']
                else:
                    row_data['x'] = cell['value']


            tags = cell['tags']
            for tag in tags:
                if tag.startswith('Ref.'):
                    row_data['Reference'] = tag
                elif tag in ('T', 'D', 'R'):
                    row_data['Confidence'] = tag
                elif tag.startswith('Estimated'):
                    if 'lower critical point' in tag:
                        row_data['Estimated LCP'] = cell['value']
                    elif 'UCP' in tag:
                        row_data['Estimated UCP'] = cell['value']

        row_datas.append(row_data)

    solute, solvent = get_compound_from_name(table['name'])
    df = pd.DataFrame(row_datas)

    df = df.replace([np.nan], [None])

    df['Solubility of:'] = solute
    df['In:'] = solvent
    # Move the solute and solvent columns to the front
    df = df[['Solubility of:', 'In:'] + [col for col in df.columns if col not in ['Solubility of:', 'In:']]]
    
    df['page'] = table['page']
    return df

def parse_all():
    dfs = []
    for source in sources:
        with pdfplumber.open(source) as pdf:
            outputs = parse_tables(pdf)
            pdf_dfs = [to_df(table) for table in outputs]
            for pdf_df in pdf_dfs:
                pdf_df['source'] = source

            dfs.extend(pdf_dfs)

    # Note this will produce a warning if some columns are all NaN.
    concat = pd.concat(dfs)
    return concat


def assign_smiles(df):
    compounds = df['Solubility of:'].unique()
    smiles = load_smiles()
    smiles_lower = {}
    for k, v in smiles.items():
        k = k.lower()
        k = k.split(';')[0]
        if k != '3-(dibutylamino)propylamine':
            k = k.split('(')[0].strip()
        smiles_lower[k] = v

    for c in compounds:
        if c is not None and c.lower() not in smiles_lower:
            print("Missing SMILES for compound", c)


    df['smiles'] = None
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        solute = row['Solubility of:']
        solvent = row['In:']
        compound = solute if solute != 'water' else solvent
        df.at[i, 'smiles'] = smiles_lower[compound.lower()]

    return df


def prepare_data_for_learning(df):
    df = df.copy()

    # drop rows where solute/solvent is None
    df = df.dropna(subset=['Solubility of:', 'In:'])

    df = assign_smiles(df)

    # Add aiw column. If 'solubility of' is water, set aiw to 0.0
    df['aiw'] = 0.0
    df.loc[df['Solubility of:'].str.lower() != 'water', 'aiw'] = 1.0
    df.loc[df['Solubility of:'].str.lower() == 'water', 'aiw'] = 0.0

    # Use smoothed values where applicable
    df['x'] = df['x (smoothed)'].combine_first(df['x'])
    df['T'] = df['T (smoothed)'].combine_first(df['T'])

    # For columns where water is the solute, x = 1 - x
    df.loc[df['Solubility of:'].str.lower() == 'water', 'x'] = 1 - df['x']

    df['x'] = df['x'].astype(np.float32)
    df['T'] = df['T'].astype(np.float32)

    # # Drop any values where confidence is marked as 'T', 'D'
    df = df[~df['Confidence'].isin(['T', 'D'])]
    return df



def compare_to_human(scraper):
    human = pd.read_csv('data/Solubility data C4-C24.csv')

    human_compounds = list(human['In:'].unique())

    def clean(s):
        s = s.lower()
        s = s.split(' ')[0]
        s = s.split('\xa0')[0]
        s = s.replace(';', '')
        return s

    human_compounds = set([clean(c) for c in human_compounds if type(c) == str])

    scraper_compounds = list(scraper['In:'].unique())
    scraper_compounds = set([clean(c) for c in scraper_compounds if type(c) == str])
    print("Human compounds:", len(human_compounds))
    print("Scraper compounds:", len(scraper_compounds))

def plot_pdf_mutual_solubility(df):
    df = df.copy()
    df['chno'] = df['smiles'].apply(smiles_to_chno)

    # Group by solute (amine) and solvent
    mutual_sol = defaultdict(lambda: {'x': [], 'T': [], 'ref': [], 'smiles': None, 'name': None, 'chno': None})
    for _, row in df.iterrows():
        solute, solvent = row['Solubility of:'], row['In:']
        smiles = row['smiles']
        x, T = row['x'], row['T']
        ref = row.get('Reference', None)
        chno = smiles_to_chno(smiles)

        if solute.lower() != 'water':
            key = (solute, solvent)
            mutual_sol[key]['name'] = solute
        else:
            key = (solvent, solute)
            mutual_sol[key]['name'] = solvent

        mutual_sol[key]['x'].append(x)
        mutual_sol[key]['T'].append(T)
        mutual_sol[key]['ref'].append(ref)
        mutual_sol[key]['smiles'] = smiles
        mutual_sol[key]['chno'] = chno

    # Group isomers
    isomer_groups = defaultdict(list)
    for val in mutual_sol.values():
        isomer_groups[val['chno']].append(val['smiles'])

    # Assign colors
    colorscale = list(reversed(pc.sequential.Turbo))
    colors = {}
    idx = 0
    for chno, smiles_list in isomer_groups.items():
        for s in smiles_list:
            if s not in colors:
                colors[s] = colorscale[idx % len(colorscale)]
                idx += 1

    # Build plot
    fig = go.Figure()
    for (name, _), val in mutual_sol.items():
        smiles = val['smiles']
        chno = val['chno']
        color = colors[smiles]
        hover = [
            f"{name}<br>x = {x:.3f}<br>T = {T:.1f} K<br>Ref: {r if r else 'N/A'}"
            for x, T, r in zip(val['x'], val['T'], val['ref'])
        ]
        fig.add_trace(go.Scatter(
            x=val['x'],
            y=val['T'],
            mode='markers',
            name=name,
            legendgroup=chno_to_string(chno),
            legendgrouptitle=dict(text=chno_to_string(chno)),
            marker=dict(size=10, color=color),
            hoverinfo='text',
            text=hover,
            visible=True if chno == CHNO(4, 11, 1, 0) else 'legendonly'
        ))

    fig.update_xaxes(title_text="Mole fraction amine", range=[0, 1])
    fig.update_yaxes(title_text="Temperature (K)")
    fig.update_layout(
        legend=dict(entrywidth=70, entrywidthmode="pixels", groupclick='toggleitem'),
        title="Mutual Solubility — PDF Dataset",
        height=700
    )
    fig.show()
