import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt # Removed as it's not used directly
import os
import re
import sys # For exit

# Import dosage parsing logic
from .dosage_parser import parse_dosage_column, DOSAGE_VALUE_COL, DOSAGE_UNIT_COL

# --- Constants ---
RAW_DATA_PATH = '../data/raw/drugage.csv'
PROCESSED_DATA_PATH = '../data/processed/processed_drugage.pkl'
NUMERIC_COLS_TO_CONVERT = ['avg_lifespan_change_percent', 'max_lifespan_change_percent', 'weight_change_percent']
DOSAGE_COL = 'dosage'
# DOSAGE_VALUE_COL = 'dosage_value' # Moved to dosage_parser
# DOSAGE_UNIT_COL = 'dosage_unit'   # Moved to dosage_parser

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
    """Imputes missing dosage values and one-hot encodes dosage units."""
    print("\n--- Imputing Missing Dosage Values & Encoding Units ---")
    if DOSAGE_VALUE_COL not in df.columns or DOSAGE_UNIT_COL not in df.columns:
        print("Skipping imputation/encoding as dosage columns were not generated.")
        return df

    df_processed = df.copy()

    # Impute numerical NaNs with median
    if df_processed[DOSAGE_VALUE_COL].isnull().any():
        median_dosage = df_processed[DOSAGE_VALUE_COL].median()
        print(f"Imputing {df_processed[DOSAGE_VALUE_COL].isnull().sum()} missing dosage values with median ({median_dosage:.4f})")
        df_processed[DOSAGE_VALUE_COL].fillna(median_dosage, inplace=True)
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

    return df_processed


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
    df_processed = impute_encode_dosage(df_parsed)
    save_data(df_processed, PROCESSED_DATA_PATH)

    print("\n--- Data Processing Script Finished ---")

if __name__ == "__main__":
    main()