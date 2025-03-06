import re
import pdfplumber
import pandas as pd
from iupac_scraper import *
import pytest


@pytest.fixture(scope="module")
def tables():
    path = "papers/c4-c6 amines.pdf"
    pdf = pdfplumber.open(path)
    return extract_tables_with_preceding_text(pdf)

def test_1(tables):
    assert clean_and_split_table(tables[0]) == []

def test_2(tables):
    # This one flips the keys midway through
    assert len(clean_and_split_table(tables[2])) == 2

def test_3(tables):
    # This one spans two pages
    cleaned = clean_and_split_table(tables[5])[0]
    assert len(cleaned['rows']) == 1

def test_4(tables):
    """This one has a superscript in it and a name change"""
    cleaned = clean_and_split_table(tables[8])

    assert cleaned[0]['name'] == 'Solubility of diethylamine in water'
    assert cleaned[1]['name'] == 'Solubility of water in diethylamine'

    # TODO: will need to deal with this.
    assert cleaned[0]['rows'][1][0]['content'] == '143.5a'
    # assert cleaned[0]['rows'][1][0]['superscript'] == 'a'

def test_5(tables):
    """This one has tentative and doubtful values"""
    cleaned = clean_and_split_table(tables[18])
    assert len(cleaned) == 2
    cell = cleaned[0]['rows'][0][1]
    assert cell['tags'] == ['D', 'Ref. 5']

    # And also multi-value cells
    assert len(cleaned[0]['rows']) == 5
    assert len(cleaned[1]['rows']) == 13

def test_6(tables):
    """This one has a multi-value cell and an exponent"""
    cleaned = clean_and_split_table(tables[60])
    assert len(cleaned) == 2

    assert len(cleaned[0]['rows']) == 13
    assert len(cleaned[1]['rows']) == 1

def test_7(tables):
    """This table shouldn't be so long..."""
    tables[52]
