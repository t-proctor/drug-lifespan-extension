import pandas as pd
import numpy as np
import seaborn as sns
import os
import re
import sys # For exit
from sklearn.preprocessing import StandardScaler # Added import

# Import dosage parsing logic
from .dosage_parser import parse_dosage_column, DOSAGE_VALUE_COL, DOSAGE_UNIT_COL

# Determine the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Assumes src is one level down from root

# --- Constants ---
# RAW_DATA_PATH = '../data/raw/drugage.csv' # Original relative path
# PROCESSED_DATA_PATH = '../data/processed/processed_drugage.pkl' # Original relative path
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'drugage.csv')
# PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_drugage.pkl')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_drugage.pkl') # Reverted to pkl
NUMERIC_COLS_TO_CONVERT = ['avg_lifespan_change_percent', 'max_lifespan_change_percent', 'weight_change_percent']
DOSAGE_COL = 'dosage'
TARGET_COL = 'avg_lifespan_change_percent' # Define target column
STRAIN_COL = 'strain' # Define strain column
TOP_K_STRAIN = 15 # Define K for strain processing

# List of columns to eventually keep for modeling based on the plan
# Note: dosage_unit columns are generated dynamically
BASE_FEATURES_TO_KEEP = [
    'compound_name', 'species', STRAIN_COL, 'gender', 'ITP', # Categorical
    DOSAGE_VALUE_COL, # Numeric dosage
    TARGET_COL # Target variable
]
# Dynamic dose columns (like 'dose_molarity') will be added later

# Configure visualization style
sns.set_theme(style="whitegrid")

# --- Helper Functions ---

def load_data(filepath: str) -> pd.DataFrame | None:
    """Loads data from a CSV file."""
    print("--- Loading Data ---")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}.")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Make sure the raw data exists.")
        sys.exit(1) # Use sys.exit for cleaner exit
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        sys.exit(1)

def display_basic_info(df: pd.DataFrame):
    """Displays basic info, descriptive stats, and initial missing values."""
    print("\n--- Basic Data Info ---")
    df.info()
    print("\n--- Descriptive Statistics (Initial) ---")
    print(df.describe(include='all'))
    print("\n--- Initial Missing Values ---")
    print(df.isnull().sum())

def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Converts specified columns to numeric, handling errors."""
    print("\n--- Data Cleaning: Numeric Conversion ---")
    print(f"Attempting to convert columns to numeric: {cols}")
    df_cleaned = df.copy() # Avoid SettingWithCopyWarning
    for col in cols:
        if col in df_cleaned.columns:
            original_missing = df_cleaned[col].isnull().sum()
            # Ensure the column is treated as string before potential conversion issues
            # Using pd.to_numeric directly often handles this, but astype(str) is safer
            df_cleaned[col] = df_cleaned[col].astype(str)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            new_missing = df_cleaned[col].isnull().sum()
            if new_missing > original_missing:
                print(f"  - Column '{col}': Introduced {new_missing - original_missing} new NaNs during conversion (were non-numeric values).")
            elif not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                 print(f"  - Warning: Column '{col}' could not be converted to numeric dtype.")
        else:
            print(f"  - Warning: Column '{col}' not found in DataFrame.")

    print("\n--- Missing Values After Numeric Conversion ---")
    # Check which columns actually exist before trying to access them
    existing_cols = [col for col in cols if col in df_cleaned.columns]
    if existing_cols:
        print(df_cleaned[existing_cols].isnull().sum())
    else:
        print("None of the specified numeric columns were found for checking missing values.")
    return df_cleaned


def impute_encode_dosage(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing dosage values with -1, one-hot encodes dosage units, and scales dosage value."""
    print("\n--- Imputing Missing Dosage Values, Encoding Units, and Scaling Value ---")
    if DOSAGE_VALUE_COL not in df.columns or DOSAGE_UNIT_COL not in df.columns:
        print("Skipping imputation/encoding/scaling as dosage columns were not generated.")
        return df

    df_processed = df.copy()

    # Impute numerical NaNs with a distinct value (-1)
    if df_processed[DOSAGE_VALUE_COL].isnull().any():
        # median_dosage = df_processed[DOSAGE_VALUE_COL].median()
        # print(f"Imputing {df_processed[DOSAGE_VALUE_COL].isnull().sum()} missing dosage values with median ({median_dosage:.4g})")
        # df_processed[DOSAGE_VALUE_COL].fillna(median_dosage, inplace=True)
        num_missing = df_processed[DOSAGE_VALUE_COL].isnull().sum()
        distinct_value = -1
        print(f"Imputing {num_missing} missing dosage values with distinct value ({distinct_value})")
        df_processed[DOSAGE_VALUE_COL].fillna(distinct_value, inplace=True)
    else:
        print("No missing dosage values to impute.")

    # Check for unexpected nulls in dosage_unit before encoding
    # Expected categories at this point: 'molarity', 'ppm', ..., 'missing', 'unknown', 'error_*'
    if df_processed[DOSAGE_UNIT_COL].isnull().any():
         print(f"Warning: Found {df_processed[DOSAGE_UNIT_COL].isnull().sum()} unexpected nulls in {DOSAGE_UNIT_COL}. Filling with 'unknown'.")
         df_processed[DOSAGE_UNIT_COL].fillna('unknown', inplace=True) # Impute unexpected nulls

    # One-hot encode the dosage unit categories
    print("One-hot encoding dosage units...")
    # dummy_na=False: Don't create a separate column for NaN units
    # We handle 'missing' and 'unknown' explicitly if needed later.
    df_processed = pd.get_dummies(df_processed, columns=[DOSAGE_UNIT_COL], prefix='dose', dummy_na=False)
    print("Columns after one-hot encoding dosage units:", df_processed.columns.tolist())

    # Scale the dosage value column
    # Scaling helps algorithms handle features with different ranges/distributions,
    # especially important here as dosage_value contains values from originally different units.
    print(f"Applying StandardScaler to '{DOSAGE_VALUE_COL}'.")
    scaler = StandardScaler()
    # Reshape is needed as scaler expects 2D array: [n_samples, n_features]
    df_processed[DOSAGE_VALUE_COL] = scaler.fit_transform(df_processed[[DOSAGE_VALUE_COL]])
    print(f"'{DOSAGE_VALUE_COL}' scaled (mean should be ~0, std dev ~1).")

    return df_processed


def finalize_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Finalizes the DataFrame for modeling according to the plan."""
    print("\n--- Finalizing DataFrame for Modeling ---")
    df_final = df.copy()

    # 1. Drop rows with missing target
    initial_rows = len(df_final)
    df_final.dropna(subset=[TARGET_COL], inplace=True)
    rows_dropped = initial_rows - len(df_final)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with missing target ('{TARGET_COL}').")

    # 2. Process 'strain' column
    print(f"Processing '{STRAIN_COL}' column: Normalizing case, imputing missing, keeping top {TOP_K_STRAIN}...")
    # Normalize case first (convert to string just in case, then lower)
    df_final[STRAIN_COL] = df_final[STRAIN_COL].astype(str).str.lower()
    # Impute missing values (which are now represented by 'nan' string after astype(str)) with 'unknown'
    df_final[STRAIN_COL].replace('nan', 'unknown', inplace=True)
    # df_final[STRAIN_COL].fillna('unknown', inplace=True) # Original fillna

    top_strains = df_final[STRAIN_COL].value_counts().nlargest(TOP_K_STRAIN).index.tolist()
    # 'unknown' should be lowercase now if it exists
    if 'unknown' not in top_strains:
        # Check if 'unknown' exists at all before trying to add
        if 'unknown' in df_final[STRAIN_COL].unique():
             top_strains.append('unknown') # Keep explicit unknown

    # Apply Top-K + 'other' logic
    df_final[STRAIN_COL] = df_final[STRAIN_COL].apply(lambda x: x if x in top_strains else 'other')
    print(f"'{STRAIN_COL}' unique values after processing:", df_final[STRAIN_COL].unique().tolist())

    # 3. Drop unnecessary columns
    # Identify all dynamically generated dose columns
    dose_cols = [col for col in df_final.columns if col.startswith('dose_')]
    cols_to_keep = BASE_FEATURES_TO_KEEP + dose_cols

    # Ensure all columns in cols_to_keep actually exist in the dataframe
    cols_to_keep = [col for col in cols_to_keep if col in df_final.columns]
    # Columns to drop are those not in the keep list
    cols_to_drop = [col for col in df_final.columns if col not in cols_to_keep]

    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} unnecessary columns:")
        # Print only a few examples if the list is long
        print("  ", cols_to_drop[:5], "..." if len(cols_to_drop) > 5 else "")
        df_final = df_final[cols_to_keep]
    else:
        print("No columns identified for dropping.")

    print(f"Final DataFrame shape for modeling: {df_final.shape}")
    print("Final columns:", df_final.columns.tolist())

    return df_final


def save_data(df: pd.DataFrame, filepath: str):
    """Saves the processed DataFrame to a pickle file."""
    print("\n--- Saving Processed Data ---")
    # Ensure the directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            # Decide if script should exit or just warn
            # sys.exit(1)

    try:
        df.to_pickle(filepath)
        # df.to_csv(filepath, index=False) # Changed to to_csv
        print(f"Processed data saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving processed data to {filepath}: {e}")
        # Consider if this should be a critical error
        # sys.exit(1)


# --- Main Execution ---
def main():
    """Main function to run the data processing pipeline."""
    df = load_data(RAW_DATA_PATH)
    if df is None:
        # Error message printed in load_data, already exited
        return

    display_basic_info(df)
    df_cleaned = clean_numeric_columns(df, NUMERIC_COLS_TO_CONVERT)
    df_parsed = parse_dosage_column(df_cleaned, DOSAGE_COL)
    df_imputed_encoded_scaled = impute_encode_dosage(df_parsed)
    df_final = finalize_for_modeling(df_imputed_encoded_scaled)
    save_data(df_final, PROCESSED_DATA_PATH)

    print("\n--- Data Processing Script Finished ---")

if __name__ == "__main__":
    main()