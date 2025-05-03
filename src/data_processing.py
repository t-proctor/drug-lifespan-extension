import pandas as pd
import seaborn as sns
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler

# Import dosage parsing logic
from dosage_parser import (
    parse_dosage_column, DOSAGE_VALUE_COL, DOSAGE_UNIT_COL
)

# Determine the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Constants ---
RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'processed', 'drug_descriptors.csv'
)
PROCESSED_DATA_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'processed', 'processed_drug_age.pkl'
)
NUMERIC_COLS_TO_CONVERT = [
    'avg_lifespan_change_percent',
    'max_lifespan_change_percent',
    'weight_change_percent'
]
DESCRIPTOR_COLS = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']
DOSAGE_COL = 'dosage'
TARGET_COL = 'avg_lifespan_change_percent'
STRAIN_COL = 'strain'
TOP_K_STRAIN = 15  # Define K for strain processing

# List of columns to eventually keep for modeling based on the plan
# Note: dosage_unit columns are generated dynamically
BASE_FEATURES_TO_KEEP = [
    'compound_name', 'species', STRAIN_COL, 'gender', 'ITP',
    DOSAGE_VALUE_COL,
    TARGET_COL
] + DESCRIPTOR_COLS
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
        sys.exit(1)
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
    df_cleaned = df.copy()
    for col in cols:
        if col in df_cleaned.columns:
            original_missing = df_cleaned[col].isnull().sum()
            # Ensure the column is treated as string
            df_cleaned[col] = df_cleaned[col].astype(str)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            new_missing = df_cleaned[col].isnull().sum()
            if new_missing > original_missing:
                print(
                    f"  - Column '{col}': Introduced "
                    f"{new_missing - original_missing} new NaNs during "
                    f"conversion (were non-numeric values)."
                )
            elif not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                print(
                    f"  - Warning: Column '{col}' could not be converted "
                    f"to numeric dtype."
                )
        else:
            print(f"  - Warning: Column '{col}' not found in DataFrame.")

    print("\n--- Missing Values After Numeric Conversion ---")
    existing_cols = [col for col in cols if col in df_cleaned.columns]
    if existing_cols:
        print(df_cleaned[existing_cols].isnull().sum())
    else:
        print(
            "None of the specified numeric columns were found for checking "
            "missing values."
        )
    return df_cleaned


def impute_encode_dosage(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing dosage values, encodes units, scales value."""
    print(
        "\n--- Imputing Missing Dosage Values, Encoding Units, and "
        "Scaling Value ---"
    )
    if DOSAGE_VALUE_COL not in df.columns or DOSAGE_UNIT_COL not in df.columns:
        print(
            "Skipping imputation/encoding/scaling as dosage columns were "
            "not generated."
        )
        return df

    df_processed = df.copy()

    # Impute numerical NaNs with a distinct value (-1)
    if df_processed[DOSAGE_VALUE_COL].isnull().any():
        num_missing = df_processed[DOSAGE_VALUE_COL].isnull().sum()
        distinct_value = -1
        print(
            f"Imputing {num_missing} missing dosage values with distinct "
            f"value ({distinct_value})"
        )
        df_processed[DOSAGE_VALUE_COL].fillna(distinct_value, inplace=True)
    else:
        print("No missing dosage values to impute.")

    # Check for unexpected nulls in dosage_unit before encoding
    if df_processed[DOSAGE_UNIT_COL].isnull().any():
        print(
            f"Warning: Found {df_processed[DOSAGE_UNIT_COL].isnull().sum()} "
            f"unexpected nulls in {DOSAGE_UNIT_COL}. Filling with 'unknown'."
        )
        df_processed[DOSAGE_UNIT_COL].fillna('unknown', inplace=True)

    # One-hot encode the dosage unit categories
    print("One-hot encoding dosage units...")
    df_processed = pd.get_dummies(
        df_processed, columns=[DOSAGE_UNIT_COL], prefix='dose', dummy_na=False
    )
    print(
        "Columns after one-hot encoding dosage units:",
        df_processed.columns.tolist()
    )

    # Scale the dosage value column
    print(f"Applying StandardScaler to '{DOSAGE_VALUE_COL}'.")
    scaler = StandardScaler()
    # Reshape is needed as scaler expects 2D array: [n_samples, n_features]
    df_processed[DOSAGE_VALUE_COL] = scaler.fit_transform(
        df_processed[[DOSAGE_VALUE_COL]]
    )
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
        print(
            f"Dropped {rows_dropped} rows with missing target "
            f"('{TARGET_COL}')."
        )

    # 2. Process 'strain' column
    print(
        f"Processing '{STRAIN_COL}' column: Normalizing case, imputing "
        f"missing, keeping top {TOP_K_STRAIN}..."
    )
    # Normalize case first
    df_final[STRAIN_COL] = df_final[STRAIN_COL].astype(str).str.lower()
    # Impute missing values (now 'nan' string) with 'unknown'
    df_final[STRAIN_COL].replace('nan', 'unknown', inplace=True)

    top_strains = df_final[STRAIN_COL].value_counts()
    top_strains = top_strains.nlargest(TOP_K_STRAIN).index.tolist()
    if 'unknown' not in top_strains:
        if 'unknown' in df_final[STRAIN_COL].unique():
            top_strains.append('unknown')  # Keep explicit unknown

    # Apply Top-K + 'other' logic
    df_final[STRAIN_COL] = df_final[STRAIN_COL].apply(
        lambda x: x if x in top_strains else 'other'
    )
    print(
        f"'{STRAIN_COL}' unique values after processing:",
        df_final[STRAIN_COL].unique().tolist()
    )

    # 3. Drop unnecessary columns
    # Identify all dynamically generated dose columns
    dose_cols = [col for col in df_final.columns if col.startswith('dose_')]
    cols_to_keep = BASE_FEATURES_TO_KEEP + dose_cols

    # Ensure all columns in cols_to_keep actually exist
    cols_to_keep = [col for col in cols_to_keep if col in df_final.columns]
    # Columns to drop are those not in the keep list
    cols_to_drop = [col for col in df_final.columns if col not in cols_to_keep]

    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} unnecessary columns:")
        # Print only a few examples if the list is long
        print(
            "  ", cols_to_drop[:5],
            "..." if len(cols_to_drop) > 5 else ""
        )
        df_final = df_final[cols_to_keep]
    else:
        print("No columns identified for dropping.")

    print(f"Final DataFrame shape for modeling: {df_final.shape}")
    print("Final columns:", df_final.columns.tolist())

    return df_final


def save_data(df: pd.DataFrame, filepath: str):
    """Saves the processed DataFrame to a pickle file."""
    print("\n--- Saving Processed Data ---")
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")

    try:
        df.to_pickle(filepath)
        print(f"Processed data saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving processed data to {filepath}: {e}")


# --- New Function for Descriptor Handling ---
def handle_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing descriptor values using median imputation."""
    print(f"\n--- Handling Missing Descriptors (Imputation Strategy) ---")
    df_processed = df.copy()
    missing_before = df_processed[DESCRIPTOR_COLS].isnull().sum()
    print("Missing descriptor values before handling:")
    print(missing_before[missing_before > 0])

    imputed_count = 0
    for col in DESCRIPTOR_COLS:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            count = df_processed[col].isnull().sum()
            print(
                f"  - Imputing {count} missing values in '{col}' with "
                f"median ({median_val:.4g})"
            )
            df_processed[col].fillna(median_val, inplace=True)
            imputed_count += count
    if imputed_count > 0:
        print(f"Imputed a total of {imputed_count} missing descriptor values.")
    else:
        print("No missing descriptor values found to impute.")

    missing_after = df_processed[DESCRIPTOR_COLS].isnull().sum().sum()
    print(f"Total missing descriptor values after handling: {missing_after}")
    print(f"DataFrame shape after handling descriptors: {df_processed.shape}")
    return df_processed


# --- Main Execution ---
def main():
    """Main function to run the data processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process drug lifespan data, imputing missing descriptors."
    )
    parser.add_argument(
        '--input_path',
        type=str,
        default=RAW_DATA_PATH,
        help=f"Path to the input CSV file (default: {RAW_DATA_PATH})"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=PROCESSED_DATA_PATH,
        help=f"Path to save the processed output PKL file (default: {PROCESSED_DATA_PATH})"
    )

    args = parser.parse_args()

    print(f"Starting data processing (Imputation strategy for descriptors)")
    print(f"Input file: {args.input_path}")
    print(f"Output file: {args.output_path}")

    df = load_data(args.input_path)
    if df is None:
        return

    display_basic_info(df)

    # Handle descriptors
    df_handled = handle_descriptors(df)

    # Continue with existing pipeline steps
    df_cleaned = clean_numeric_columns(df_handled, NUMERIC_COLS_TO_CONVERT)
    df_parsed = parse_dosage_column(df_cleaned, DOSAGE_COL)
    df_imputed_encoded_scaled = impute_encode_dosage(df_parsed)
    df_final = finalize_for_modeling(df_imputed_encoded_scaled)

    save_data(df_final, args.output_path)

    print("\n--- Data Processing Script Finished ---")


if __name__ == "__main__":
    main()
