import re
import pdfplumber
import pandas as pd

sources = [
    "papers/c4-c6 amines.pdf",
    "papers/c7-c24 amines.pdf",
    # "papers/non-aliphatic amines.pdf",
]

def load():
    return pdfplumber.open(sources[0])


def extract_tables_with_preceding_text_from_page(page):
    """Extract multiple tables from a single PDF page and associate them with their preceding lines of text using bounding boxes."""
    tables = []
    lines = page.extract_text_lines()
    found_tables = page.find_tables()

    # Omit all lines which are inside one of the table bounding boxes
    lines = [line for line in lines if not any(table.bbox[0] < line["x0"] < table.bbox[2] and table.bbox[1] < line["top"] < table.bbox[3] for table in found_tables)]
    
    for table in found_tables:
        table_bbox = table.bbox
        preceding_text = "Unknown"
        
        for line in lines:
            line_bbox = (line["x0"], line["top"], line["x1"], line["bottom"])
            if line_bbox[3] < table_bbox[1]:  # Line is above the table
                preceding_text = line["text"]
            else:
                break
        
        extracted_table = table.extract()
        
        if extracted_table:
            cleaned_table = [row for row in extracted_table if any(cell.strip() for cell in row if cell)]
            tables.append((preceding_text, cleaned_table))
    
    return tables



def extract_sections(pdf):
    """Extract text from the PDF and break it into sections based on headers (e.g., 2.3)."""
    sections = {}
    current_section = "Unknown"
    
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            lines = text.split("\n")
            for line in lines:
                section_match = re.match(r"^(\d+\.\d+)\s+(.+)", line)
                if section_match:
                    current_section = section_match.group(1) + " " + section_match.group(2)
                    sections[current_section] = []
                if current_section in sections:
                    sections[current_section].append(line)
    
    return sections


def extract_tables_with_preceding_text_from_page(page, previous_page=None):
    """Extract multiple tables from a single PDF page and associate them with their preceding lines of text using bounding boxes."""
    tables = []
    lines = page.extract_text_lines()
    found_tables = page.find_tables()
    
    # If there is a previous page and a table at the top, check the last line of the previous page
    last_prev_line = None
    if previous_page:
        prev_lines = previous_page.extract_text_lines()
        if prev_lines:
            last_prev_line = prev_lines[-1]["text"]  # Get last line of previous page

    # Omit all lines which are inside one of the table bounding boxes
    lines = [line for line in lines if not any(table.bbox[0] < line["x0"] < table.bbox[2] and table.bbox[1] < line["top"] < table.bbox[3] for table in found_tables)]
    
    for table in found_tables:
        table_bbox = table.bbox
        preceding_text = "Unknown"
        
        # If the table is at the top of the page and there's a previous page, use last line from previous page
        if table_bbox[1] < 0.15 * page.height and last_prev_line:
            preceding_text = last_prev_line
        else:
            for line in lines:
                line_bbox = (line["x0"], line["top"], line["x1"], line["bottom"])
                if line_bbox[3] < table_bbox[1]:  # Line is above the table
                    preceding_text = line["text"]
                else:
                    break
        
        extracted_table = table.extract()
        
        if extracted_table:
            cleaned_table = [row for row in extracted_table if any(cell.strip() for cell in row if cell)]
            tables.append({"preceding_text": preceding_text, "table": cleaned_table, "bbox": table_bbox})
    
    return tables


def extract_tables_with_preceding_text(pdf):
    """Extract tables from the given PDF file page by page, associating them with preceding text and handling multi-page tables more cleanly."""
    all_tables = []
    previous_table = None
    previous_page_text = []
    
    for i, page in enumerate(pdf.pages):
        tables = []
        lines = page.extract_text_lines()
        found_tables = page.find_tables()
        
        # Identify topmost and bottommost objects
        all_objects = lines + found_tables
        all_objects.sort(key=lambda obj: obj["top"] if isinstance(obj, dict) else obj.bbox[1])
        topmost_object = all_objects[0] if all_objects else None
        bottommost_object = all_objects[-1] if all_objects else None
        
        # Use previously saved text for multi-page table headings
        last_prev_lines = previous_page_text[-2:] if len(previous_page_text) >= 2 else previous_page_text
        previous_page_text = []  # Reset for next iteration
        
        for table in found_tables:
            table_bbox = table.bbox
            preceding_text = "Unknown"
            
            # Use saved text if the table is the first object on the current page
            if table == topmost_object and last_prev_lines:
                preceding_text = "\n".join(last_prev_lines)
            else:
                preceding_text_candidates = [line["text"] for line in lines if line["bottom"] < table_bbox[1]]
                preceding_text = "\n".join(preceding_text_candidates[-2:]) if len(preceding_text_candidates) >= 2 else preceding_text_candidates[0] if preceding_text_candidates else "Unknown"

            # Slightly expand y_tolerance so that superscripts will not be relocated to newlines            
            extracted_table = table.extract(y_tolerance=4)
            
            if extracted_table:
                cleaned_table = [row for row in extracted_table if any(cell.strip() for cell in row if cell)]
                tables.append({"preceding_text": preceding_text, "table": cleaned_table, "bbox": table_bbox})
        
        # Save non-table text for the next page
        previous_page_text = [line["text"] for line in lines if not any(table.bbox[1] < line["top"] < table.bbox[3] for table in found_tables)]
        
        # Merge multi-page tables more cleanly
        for table in tables:
            if previous_table and len(table["table"]) > 0:
                if previous_table["bbox"][3] > 0.9 * page.height and table["bbox"][1] < 0.1 * page.height:
                    previous_table["table"].extend(table["table"])
                else:
                    all_tables.append(previous_table)
                    previous_table = table
            else:
                previous_table = table
    
    if previous_table:
        all_tables.append(previous_table)
    
    return all_tables

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

    # We want to handle newlines which can occur when there are multiple values.
    # However, we also need to handle exponents specially first.
    if '\n' in cell_with_comma:
        if '× 10' in cell_with_comma:
            split = cell_with_comma.split('\n', 1)
            exps_str = split[0]
            exps = exps_str.split(' ')
            rest = split[1]



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
        return { 'content': '', 'tags': [], 'superscript': superscript, 'source': source }

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
                    print("Stopping parsing after got", row)
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

    parsed = []
    for table in tables:
        parsed.extend(clean_and_split_table(table))

    return parsed

import re

def parse_scientific_notation(s):
    """Parses a string in the format '9.24 × 10^−3' and returns a float."""
    match = re.match(r"([\d\.]+)\s*×\s*10\^([−\-]?\d+)", s)
    if not match:
        raise ValueError(f"Invalid scientific notation format: {s}")
    
    base, exponent = match.groups()
    exponent = exponent.replace('−', '-')  # Handle minus signs
    return float(base) * (10 ** int(exponent))


def transform_row(row, keys):
    # Handle exponents and references
    update = []
    for cell in row:
        if cell is None:
            # TODO: deal with this - parsing issue
            update.append(None)
            continue
            
        # Handle cases like: -3\n2.293 x 10 with regex
        cell = cell.strip()
        if '\n' in cell:
            exp, n = cell.split('\n')
            # if exp is a digit
            # check if exp is a digit
            if exp.isdigit():
                cell = f"{n}^{(exp)}"
                cell = parse_scientific_notation(cell)
            else:
                # TODO: handle footnotes / references
                cell = n
        if '×' in cell:
            first, second = cell.split('×')
            cell = handle_plus_minus(first.strip())
            if second.strip() == '10':
                cell = handle_plus_minus(first.strip()) * 10

        if type(cell) == str and '±' in cell:
            cell = handle_plus_minus(cell)

        update.append(cell)
    return update

def handle_plus_minus(str):
    out = str.replace('(', '').replace(')', '')
    return float(out.split('±')[0])

def parse_tables(pdf):
    tables = organize_tables(pdf)
    for table in tables:
        keys = table['keys']
        # table['rows'] = [transform_row(row, keys) for row in table['rows']]

    return tables


def parse_all():
    outputs = []
    for source in sources:
        with pdfplumber.open(source) as pdf:
            outputs += parse_tables(pdf)

    print(len(outputs))
    print(sum([len(x['rows']) for x in outputs]), 'points')
    return outputs