import pandas as pd
import requests
import time

import re

def clean_name_for_pubchem(name):
    # Strip trailing parenthetical if the *entire* thing is at the end of the name
    if re.match(r'^.+ \([^\(\)]+\)$', name):
        return name[:name.rindex('(')].strip()
    return name.strip()


def get_smiles_from_pubchem(name):
    name = clean_name_for_pubchem(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return smiles
    except Exception as e:
        print(f"Error retrieving SMILES for {name}: {e}")
        return None

def update_csv_with_smiles(input_csv, output_csv='data/amine_smiles_pubchem.csv'):
    df = pd.read_csv(input_csv)
    if 'Compound Name' not in df.columns:
        print("Input CSV must have a 'name' column.")
        return

    df['Smiles (PubChem)'] = None

    for idx, row in df.iterrows():
            name = row['Compound Name']
            smiles = get_smiles_from_pubchem(name)
            print('got smiles for', name)
            df.at[idx, 'Smiles (PubChem)'] = smiles or "NOT FOUND"
            time.sleep(0.2)  # Be polite to the API

    df.to_csv(output_csv, index=False)
    return df

# Example usage:
# update_csv_with_smiles('compounds.csv', 'compounds_with_smiles.csv')
