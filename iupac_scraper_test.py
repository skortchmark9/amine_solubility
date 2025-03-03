import re
import pdfplumber
import pandas as pd
from iupac_scraper import *

path = "papers/c4-c6 amines.pdf"
pdf = pdfplumber.open(path)

def test_1(table):
    assert clean_and_split_table(table) == []

def test_2(table):
    # This one flips the keys midway through
    assert len(clean_and_split_table(table)) == 2

def test_3(table):
    # This one spans two pages
    cleaned = clean_and_split_table(table)[0]
    assert len(cleaned['rows']) == 1

def test_4(table):
    """This one has a superscript in it and a name change"""
    cleaned = clean_and_split_table(table)

    assert cleaned[0]['name'] == 'Solubility of diethylamine in water'
    assert cleaned[1]['name'] == 'Solubility of water in diethylamine'
    assert cleaned[0]['rows'][1][0]['superscript'] == 'a'

def test_5(table):
    """This one has tentative and doubtful values"""
    cleaned = clean_and_split_table(table)
    assert len(cleaned) == 2
    cell = cleaned[0]['rows'][0][1]
    assert cell['tags'] == ['D', 'Ref. 5']

def main():
    tables = extract_tables_with_preceding_text(pdf)
    test_1(tables[0])
    test_2(tables[2])
    test_3(tables[5])
    test_4(tables[8])
    test_5(tables[18])
    print('All tests passed.')


if __name__ == '__main__':
    main()