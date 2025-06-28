from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import Draw

# Function to calculate Tanimoto similarity using Morgan Fingerprints
def calculate_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None # Indicate invalid SMILES

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    # Calculate the Tanimoto similarity between the two fingerprints
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

# Function to calculate molecular descriptors (LogP, Molecular Weight, etc.)
def calculate_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # Indicate invalid SMILES

    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'TPSA': CalcTPSA(mol),  # Topological Polar Surface Area
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
    }
    return descriptors

# Function to compare functional groups using SMARTS patterns
def compare_functional_groups(smiles1, smiles2):
    patterns = {
        'Carboxylic Acid': '[CX3](=O)[O;H]',
        'Hydroxyl Group': '[OX2H]',
        'Aromatic Ring': 'a',
        'Amine': '[NX3;H2,H1;!$(NC=O)]',
    }

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None # Indicate invalid SMILES

    functional_groups1 = {group: len(mol1.GetSubstructMatches(Chem.MolFromSmarts(pattern))) for group, pattern in patterns.items()}
    functional_groups2 = {group: len(mol2.GetSubstructMatches(Chem.MolFromSmarts(pattern))) for group, pattern in patterns.items()}

    # Count common functional groups
    common_groups = sum(min(functional_groups1[group], functional_groups2[group]) for group in patterns)
    return common_groups

# Function to generate 2D structure images
def generate_mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(200, 200))
    # To save the image to a bytes buffer, which can be sent to Flask
    from io import BytesIO
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr