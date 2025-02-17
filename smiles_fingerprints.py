import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Create a Morgan fingerprint generator with desired parameters


def create_morgan_generator(radius=0, nBits=10):
    cache = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)

    def get_morgan_fingerprint(smiles):
        if smiles in cache:
            return cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate fixed-length fingerprint
        fingerprint = generator.GetCountFingerprintAsNumPy(mol)
        cache[smiles] = fingerprint
        return fingerprint

    return get_morgan_fingerprint



# Create a Morgan fingerprint generator with desired parameters
default_morgan_generator = create_morgan_generator(radius=0, nBits=10)

def get_morgan_fingerprint(smiles):
    return default_morgan_generator(smiles)

def count_unique_indexes(df):
    """Count the number of unique indexes in the fingerprint column."""
    unique_indexes = set()
    for i, row in df.iterrows():
        unique_indexes.update(row['Solute Fingerprint'])


def load_smiles():
    """Simplified Molecular Input Line Entry System (SMILES)
    codes for each compound."""
    df = pd.read_csv('data/amine_smiles.csv')
    smiles_map = {}
    for i, row in df.iterrows():
        smiles_map[row['Compound Name']] = row['SMILES Code']
    return smiles_map
