import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import MolsToGridImage
from IPython.display import display
from PIL import Image
import io
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG


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
default_morgan_generator = create_morgan_generator(radius=2, nBits=2048)

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


def compute_complexity(mol):
    return Descriptors.BertzCT(mol)



def compute_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES

    num_rings = Descriptors.RingCount(mol)
    try:
        features = {
            'molecular_weight_gpm': Descriptors.MolWt(mol),
            'logP': Crippen.MolLogP(mol),  # Not XLogP3-AA, but standard logP from RDKit
            'hydrogen_bond_donor_count': Lipinski.NumHDonors(mol),
            'hydrogen_bond_acceptor_count': Lipinski.NumHAcceptors(mol),
            'rotatable_bond_count': Lipinski.NumRotatableBonds(mol),
            'topological_polar_surface_area_angstroms': rdMolDescriptors.CalcTPSA(mol),
            'complexity': compute_complexity(mol),
            'undefined_atom_stereocenter_count': len([
                center for center in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                if center[1] == '?'
            ]),
            'num_rings':   num_rings,
            'fsp3':        Descriptors.FractionCSP3(mol),
            'aromatic_prop': Descriptors.NumAromaticRings(mol) / num_rings if num_rings > 0 else 0.0,
        }
        return features
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def draw_fingerprint_bit_combined(smiles, bit_id, radius=2, n_bits=2048):
    """
    Draw the full molecule with all substructures contributing to a fingerprint bit highlighted.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    
    smiles_map = load_smiles()
    # Lookup name in smiles_map
    name = None
    for k, v in smiles_map.items():
        if v == smiles:
            name = k
            break

    Compute2DCoords(mol)

    bitInfo = {}
    _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, bitInfo=bitInfo
    )

    if bit_id not in bitInfo:
        raise ValueError(f"Bit {bit_id} didn’t fire for this molecule.")

    # Collect all atoms and bonds involved in this bit
    highlight_atoms = set()
    highlight_bonds = set()

    for atom_idx, rad in bitInfo[bit_id]:
        env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx))
        highlight_bonds.update(env_bonds)
        for bidx in env_bonds:
            bond = mol.GetBondWithIdx(bidx)
            highlight_atoms.add(bond.GetBeginAtomIdx())
            highlight_atoms.add(bond.GetEndAtomIdx())
        highlight_atoms.add(atom_idx)

    # Prepare drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=f"Bit {bit_id} for {name} ({smiles})",  # ← your label here
        highlightAtoms=list(highlight_atoms),
        highlightBonds=list(highlight_bonds),
        highlightAtomColors={i: (0.9, 0.3, 0.3) for i in highlight_atoms},
        highlightBondColors={i: (0.9, 0.3, 0.3) for i in highlight_bonds},
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    img = Image.open(io.BytesIO(png))
    return img


def find_molecule_with_bit_on(bit_id=981, radius=2, n_bits=2048):
    smiles_map = load_smiles()
    for name, smiles in smiles_map.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bitInfo = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bitInfo)
        if bit_id in bitInfo:
            print(f"Found bit {bit_id} in molecule {name}: {smiles}")
            return smiles, bitInfo
    print(f"No molecule found with bit {bit_id}")
    return None, None
