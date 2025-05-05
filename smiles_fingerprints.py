import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import MolsToGridImage
from IPython.display import display


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


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def extract_substructures_for_bit(smiles, bit_id, radius=2, n_bits=2048):
    # 1) Parse molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # 2) Compute fingerprint + bit‐info
    bitInfo = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
        bitInfo=bitInfo
    )
    # bitInfo: {bit_id: [(atom_idx, radius), ...], ...}

    if bit_id not in bitInfo:
        return []  # this bit never fires for this mol

    substructures = []
    for atom_idx, rad in bitInfo[bit_id]:
        # 3) Get the atom environment
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)

        # 4) Highlight the central atom
        #    so you can see which neighborhood you're looking at:
        highlight = [amap[atom_idx]]

        # 5) Turn it into a SMARTS (or SMILES)
        smarts = Chem.MolToSmarts(submol, rootedAtAtom=highlight[0])
        smi    = Chem.MolToSmiles(submol, rootedAtAtom=highlight[0])

        substructures.append({
            'atom_index': atom_idx,
            'radius': rad,
            'smarts': smarts,
            'smiles': smi
        })

    return substructures

def draw_fingerprint_bit(smiles, bit_id, radius=2, n_bits=2048):
    """
    For the given SMILES and Morgan fingerprint bit index,
    returns a list of PIL images where each image highlights
    one atom‐environment that contributed to that bit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    # Compute ECFP fingerprint, capturing bitInfo
    bitInfo = {}
    _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, bitInfo=bitInfo
    )
    if bit_id not in bitInfo:
        raise ValueError(f"Bit {bit_id} didn’t fire for this molecule.")
    
    images = []
    for atom_idx, rad in bitInfo[bit_id]:
        # find all bonds in the radius‐N environment
        env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx))
        # collect all atoms in that environment
        atom_set = {atom_idx}
        for bidx in env_bonds:
            bond = mol.GetBondWithIdx(bidx)
            atom_set.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        # draw with highlights
        img = Draw.MolToImage(
            mol,
            highlightAtoms=list(atom_set),
            highlightBonds=env_bonds,
            size=(300, 300)
        )
        images.append((rad, img))
    return images
