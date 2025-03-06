import re
import pdfplumber
import pandas as pd

sources = [
    "papers/c4-c6 amines.pdf",
    # "papers/c7-c24 amines.pdf",
    # "papers/non-aliphatic amines.pdf",
]

def load():
    return pdfplumber.open(sources[0])


def partition(x, cond):
    return [x for x in x if cond(x)], [x for x in x if not cond(x)]

def extract_tables_with_preceding_text(pdf):
    tables_with_text = []

    for page in pdf.pages:
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
        elif prev_table and table['is_top'] and prev_table['is_bottom']:
            prev_table['table'].extend(table['table'])
            prev_table['lines_below'] = table['lines_below']
            prev_table['is_bottom'] = table['is_bottom']
            continue
        elif prev_table and prev_table['lines_below']:
            preceding_lines = prev_table['lines_below'][:2]

        out.append({
            'preceding_text': '\n'.join(preceding_lines),
            'table': table['table'],
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

    return {
        'content': content,
        'tags': tags,
        'superscript': superscript,
        'source': source,
        'value': None,
    }

def likely_number(s):
    # Check that it contains only digits, decimal point, x, and ±
    return all(c.isdigit() or c in "\n ,.×±-" for c in s)

def likely_key(s):
    return any([
        'compiler' in s,
        'Solubility' in s,
        'w1' in s,
        'w2' in s,
        'T/' in s,
        't/' in s,
        'x2' in s,
        'Smoothed' in s,
        'Experimental values' in s,
    ])


def clean_and_split_table(table):
    out = []
    header = table["preceding_text"]

    if header.lower().startswith("experimental values") or re.match("^Table \\d+\\.", header, re.IGNORECASE):
        if '+' in header:
            return out

        # print(header)
        name = header.split('\n')[1]
        keys = table['table'][0]
        input_rows = table['table'][1:]
        output_rows = []

        def finish():
            out.append({
                'name': name,
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
                    # Parsing messed up somehow
                    print("Stopping parsing after got", header, row)
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
    print(table['name'])
    print(table['keys'])
    print(len(table['rows']), 'rows')
    if table['rows']:
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
        raise ValueError(f"Invalid scientific notation format: {s}")
    
    base, exponent = match.groups()
    exponent = exponent.replace('−', '-')  # Handle minus signs
    return float(base) * (10 ** int(exponent))

def is_number_with_ending_superscript(s):
    """Checks if a string is a number with a superscript at the end."""
    return re.match(r"[\d\.]+[a-z]", s)

def is_exponent(s):
     return '×' in s

def transform_row(row, keys):
    # Handle exponents and references
    update = []
    for cell in row:
        cell = cell.copy()
        if is_number_with_ending_superscript(cell['content']):
            cell['superscript'] = cell['content'][-1]
            cell['value'] = float(cell['content'][:-1])
        elif is_exponent(cell['content']):
            cell['value'] = parse_scientific_notation(cell['content'])
        elif '±' in cell['content']:
            cell['value'] = handle_plus_minus(cell['content'])
        elif cell['content']:
            cell['value'] = float(cell['content'])

        update.append(cell)
    return update

def handle_plus_minus(str):
    out = str.replace('(', '').replace(')', '')
    return float(out.split('±')[0])

def parse_tables(pdf):
    tables = organize_tables(pdf)
    for table in tables:
        keys = table['keys']
        try:
            table['rows'] = [transform_row(row, keys) for row in table['rows']]
        except Exception as e:
            print(table)
            raise e
            

    return tables


def parse_all():
    outputs = []
    for source in sources:
        with pdfplumber.open(source) as pdf:
            outputs += parse_tables(pdf)

    print(len(outputs))
    print(sum([len(x['rows']) for x in outputs]), 'points')
    return outputs