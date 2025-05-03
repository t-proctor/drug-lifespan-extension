import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import pubchempy as pcp
from tqdm import tqdm
import time
import os


def get_smiles_from_name(name):
    """Attempts to retrieve canonical SMILES from PubChem
        using compound name."""
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            return compounds[0].canonical_smiles
    except Exception as e:
        print(f"Error retrieving SMILES for '{name}': {e}")
    return None


def calculate_descriptors(smiles):
    """Calculates desired descriptors from RDKit."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        molwt = Descriptors.MolWt(mol)
        hdonors = Descriptors.NumHDonors(mol)
        hacceptors = Descriptors.NumHAcceptors(mol)
        return {
            'LogP': logp, 'TPSA': tpsa, 'MolWt': molwt,
            'NumHDonors': hdonors, 'NumHAcceptors': hacceptors
        }
    except Exception as e:
        print(f"Error calculating descriptors for SMILES '{smiles}': {e}")
        return None


def main():
    input_csv = 'data/raw/drugage.csv'
    output_csv = 'data/processed/drug_descriptors.csv'

    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if 'compound_name' not in df.columns:
        print(f"Error: 'compound_name' column not found in {input_csv}")
        return

    unique_compounds = df['compound_name'].unique()
    print(f"Found {len(unique_compounds)} unique compounds.")

    descriptor_cache = {}
    failed_lookups = []

    print("Fetching SMILES from PubChem and calculating descriptors...")
    for name in tqdm(unique_compounds, desc="Processing compounds"):
        if pd.isna(name):
            continue
        smiles = get_smiles_from_name(name)
        if smiles:
            descriptors = calculate_descriptors(smiles)
            if descriptors:
                descriptor_cache[name] = descriptors
            else:
                print(
                    f"Could not calculate descriptors for '{name}' "
                    f"(SMILES: {smiles})"
                )
                failed_lookups.append(name)
        else:
            print(f"Could not find SMILES for '{name}' on PubChem.")
            failed_lookups.append(name)

        # Add a small delay to avoid overwhelming PubChem API
        time.sleep(0.1)

    print(f"\nSuccessfully processed {len(descriptor_cache)} compounds.")
    if failed_lookups:
        failed_str = ", ".join(failed_lookups[:10])
        ellipsis = "..." if len(failed_lookups) > 10 else ""
        print(
            f"Failed to process {len(failed_lookups)} compounds: "
            f"{failed_str}{ellipsis}"
        )

    # Create new columns in the DataFrame
    descriptor_cols = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']
    for col in descriptor_cols:
        df[col] = df['compound_name'].map(
            lambda name: descriptor_cache.get(name, {}).get(col, None)
        )

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"Saving data with descriptors to {output_csv}...")
    try:
        df.to_csv(output_csv, index=False)
        print("Processing complete.")
    except Exception as e:
        print(f"Error saving output CSV: {e}")


if __name__ == "__main__":
    main()
