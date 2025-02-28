import re
import pdfplumber
import pandas as pd

sources = [
    "papers/c4-c6 amines.pdf",
    # "papers/c7-c24 amines.pdf",
    # "papers/non-aliphatic amines.pdf",
]

def load():
    return pdfplumber.open(pdf_path)


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



# def extract_tables_with_preceding_text(pdf):
#     """Extract tables from the given PDF file page by page, handling multiple tables per page and merging tables across pages when necessary."""
#     all_tables = []
#     previous_table = None
    
#     for page in pdf.pages:
#         tables = extract_tables_with_preceding_text_from_page(page)
        
#         for table in tables:

#             # Continuation of previous table
#             if previous_table and len(table["table"]) > 0:
#                 # Check if table starts without headers (i.e., appears to be a continuation)
#                 if all(cell.isdigit() or cell.replace('.', '', 1).isdigit() for cell in table["table"][0] if cell):
#                     previous_table["table"].extend(table["table"])  # Append to previous
#                 else:
#                     all_tables.append(previous_table)  # Store previous
#                     previous_table = table  # Start a new table
#             else:
#                 previous_table = table
    
#     if previous_table:
#         all_tables.append(previous_table)  # Store last table
    
#     return all_tables
import pdfplumber
import pandas as pd
import re

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
            
            extracted_table = table.extract()
            
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


def organize_tables(pdf):
    tables = extract_tables_with_preceding_text(pdf)

    parsed = []
    for table in tables:
        header = table["preceding_text"]
        if header.startswith("Experimental Values") or re.match("^Table \\d+\\.$", header):
            name = header.split('\n')[1]
            keys = table['table'][0]
            rows = []

            i = 1
            while i < len(table['table']):
                row = table['table'][i]
                # Messed up tables

                # Rows can flip in the middle of the table, so create a new one in that case
                if row[0].lower().startswith('solubility of'):
                    parsed.append({
                        'name': name,
                        'keys': keys,
                        'rows': rows
                    })
                    name = row[0]
                    rows = []
                    keys = table['table'][i+1]
                    i += 2

                if len(row) != len(keys):
                    # Parsing messed up somehow
                    print("Stopping parsing after got", row)
                    break

                rows.append(row)
                i += 1

            if rows:
                parsed.append({
                    'name': name,
                    'keys': keys,
                    'rows': rows
                })

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
        table['rows'] = [transform_row(row, keys) for row in table['rows']]

    return tables


def parse_all():
    outputs = []
    for source in sources:
        with pdfplumber.open(source) as pdf:
            outputs += parse_tables(pdf)

    print(len(outputs))
    print(sum([len(x['rows']) for x in outputs]))