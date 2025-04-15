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
            print(smiles, 'problem')
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

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG


def visualize_fp_bit(smiles, bit_id, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Generate fingerprint with bitInfo
    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo)
    print(bitInfo)

    # Get atoms that contributed to this bit
    if bit_id not in bitInfo:
        print(f"Bit {bit_id} not found in molecule.")
        return

    # Highlight atoms from the first matching environment
    atom_id, rad = bitInfo[bit_id][0]
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_id)
    atoms = set()
    for bidx in env:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())

    # Draw with highlights
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(atoms))
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return SVG(svg)



def visualize_fp_bit2(smiles: str, bit: int, radius: int = 2, nBits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    bitInfo = {}
    _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo)

    if bit not in bitInfo:
        raise ValueError(f"Bit {bit} not found in this molecule")

    # There may be multiple atom/radius pairs that set the bit — pick the first one
    atom_idx, rad = bitInfo[bit][0]

    return Draw.DrawMorganBit(mol, bit, (atom_idx, rad), useSVG=False)
