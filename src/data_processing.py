import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt # Removed as it's not used directly
import os
import re
import sys # For exit

# --- Constants ---
RAW_DATA_PATH = '../data/raw/drugage.csv'
PROCESSED_DATA_PATH = '../data/processed/processed_drugage.pkl'
NUMERIC_COLS_TO_CONVERT = ['avg_lifespan_change_percent', 'max_lifespan_change_percent', 'weight_change_percent']
DOSAGE_COL = 'dosage'
DOSAGE_VALUE_COL = 'dosage_value'
DOSAGE_UNIT_COL = 'dosage_unit'

# --- Dosage Parsing Constants ---
MOLARITY_RE = re.compile(
    r"(?P<value>\d+\.?\d*)\s*(?P<prefix>[pnuµm]?)(?P<unit>M)\\b",
    re.IGNORECASE
)
MOLARITY_PREFIX_FACTORS = {
    "p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0
}
# Order matters: More specific patterns first might be better, but using original order.
# Consider reviewing pattern overlaps if needed.
DOSAGE_PATTERNS = {
    "ppm":           re.compile(r"(?P<value>\d+\.?\d*)\s*ppm", re.IGNORECASE),
    "percent":       re.compile(r"(?P<value>\d+\.?\d*)\s*%", re.IGNORECASE),
    "µg_per_ml":     re.compile(r"(?P<value>\d+\.?\d*)\s*µ?g/ml", re.IGNORECASE),
    "mg_per_ml":     re.compile(r"(?P<value>\d+\.?\d*)\s*mg/ml", re.IGNORECASE),
    "mg_per_100ml":  re.compile(r"(?P<value>\d+\.?\d*)\s*mg/100\s*m[l|L]", re.IGNORECASE),
    "mg_per_kg":     re.compile(r"(?P<value>\d+\.?\d*)\s*mg/kg", re.IGNORECASE),
    "mg_per_L":      re.compile(r"(?P<value>\d+\.?\d*)\s*mg/l", re.IGNORECASE),
    "µl_per_100ml":  re.compile(r"(?P<value>\d+\.?\d*)\s*µ?l/100\s*m[l|L]", re.IGNORECASE),
    "g_per_L":       re.compile(r"(?P<value>\d+\.?\d*)\s*g/l", re.IGNORECASE),
    "µg_per_g":      re.compile(r"(?P<value>\d+\.?\d*)\s*µ?g/g", re.IGNORECASE),
    "iu":            re.compile(r"(?P<value>\d+\.?\d*)\s*iu", re.IGNORECASE),
    "ng_per_ml":     re.compile(r"(?P<value>\d+\.?\d*)\s*ng/ml", re.IGNORECASE),
}

# Configure visualization style
sns.set(style="whitegrid")

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


def _parse_molarity(s: str) -> tuple[float | None, str | None]:
    """Helper to parse molarity strings (e.g., '10 µM'). Returns standardized value in M and original unit string."""
    if not isinstance(s, str): return None, None
    try:
        s_cleaned = s.replace("μ", "µ").strip() # Normalize mu
    except AttributeError: return None, None # Should not happen with isinstance check

    m = MOLARITY_RE.search(s_cleaned)
    if not m: return None, None

    try:
        val = float(m.group("value"))
        pfx = m.group("prefix").lower() if m.group("prefix") else ""
        unit_str = pfx + m.group("unit").upper() # e.g., µM, mM, M
        std_value = val * MOLARITY_PREFIX_FACTORS.get(pfx, 1.0) # Value in base M
        return std_value, unit_str
    except (ValueError, AttributeError, TypeError):
        # Log or handle error if needed, e.g., invalid number format after regex match
        return None, None


def _parse_other_dosage(s: str) -> tuple[float | None, str | None]:
    """Helper to parse non-molarity dosage strings based on defined patterns."""
    if not isinstance(s, str): return None, None
    try:
        # Normalize: replace unicode mu, strip, remove parenthetical footnotes
        txt = s.replace("μ", "µ").strip()
        txt = re.sub(r"\s*\(.*?\)\s*", "", txt).strip()
    except AttributeError:
        return None, "error_normalizing" # Or return None, None

    for unit_category, pattern in DOSAGE_PATTERNS.items():
        m = pattern.search(txt)
        if m:
            try:
                return float(m.group("value")), unit_category
            except (ValueError, IndexError):
                 return None, f"error_extracting_{unit_category}" # Or return None, None
    return None, None


def parse_dosage_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Parses the dosage column into numeric value and unit category."""
    print("\n--- Comprehensive Dosage Parsing ---")
    if col_name not in df.columns:
        print(f"Warning: '{col_name}' column not found. Skipping parsing.")
        return df

    df_parsed = df.copy()

    def parse_single_dosage(s):
        # 1. Handle NaN/empty input
        if pd.isna(s) or (isinstance(s, str) and not s.strip()):
            return np.nan, "missing"

        # 2. Try molarity parser (returns standardized value in M)
        std_molar_value, _ = _parse_molarity(str(s)) # Ensure input is string
        if std_molar_value is not None:
            return std_molar_value, "molarity" # Unit category is 'molarity'

        # 3. Try other parsers
        other_value, unit_category = _parse_other_dosage(str(s)) # Ensure input is string
        if other_value is not None:
            if unit_category and "error" in unit_category:
                 return np.nan, unit_category # Pass through specific error code
            else:
                 return other_value, unit_category # Successfully parsed non-molarity unit

        # 4. If nothing matched, mark as unknown
        return np.nan, "unknown"

    # Apply the combined parser
    parsed_results = df_parsed[col_name].astype(str).apply(parse_single_dosage)
    df_parsed[[DOSAGE_VALUE_COL, DOSAGE_UNIT_COL]] = pd.DataFrame(parsed_results.tolist(), index=df_parsed.index)

    # --- Report Summary Statistics ---
    total_entries = len(df_parsed)
    parsed_ok_mask = df_parsed[DOSAGE_VALUE_COL].notnull()
    parsed_count = parsed_ok_mask.sum()
    missing_explicit_mask = (df_parsed[DOSAGE_UNIT_COL] == 'missing')
    missing_explicit_count = missing_explicit_mask.sum()
    unknown_mask = (df_parsed[DOSAGE_UNIT_COL] == 'unknown')
    unknown_count = unknown_mask.sum()
    # Identify errors (dosage_value is NaN but unit is not 'missing' or 'unknown')
    error_mask = (~parsed_ok_mask & ~missing_explicit_mask & ~unknown_mask)
    error_count = error_mask.sum()

    print("\n--- Dosage Parsing Results ---")
    print(f"Total entries: {total_entries}")
    print(f" -> Explicitly missing (original NaN): {missing_explicit_count} ({missing_explicit_count/total_entries*100:.2f}%)")
    print(f" -> Successfully parsed (value extracted): {parsed_count} ({parsed_count/total_entries*100:.2f}%)")
    print(f" -> Unmatched/Unknown format: {unknown_count} ({unknown_count/total_entries*100:.2f}%)")
    if error_count > 0:
        print(f" -> Errors during parsing/extraction: {error_count} ({error_count/total_entries*100:.2f}%)")
        # Optional: View specific errors
        # print(df_parsed.loc[error_mask, [col_name, DOSAGE_UNIT_COL]].head())

    print("\n--- Dosage Unit Distribution (Successfully Parsed) ---")
    # Exclude 'missing', 'unknown', and errors from value counts of units
    unit_counts = df_parsed.loc[parsed_ok_mask, DOSAGE_UNIT_COL].value_counts(normalize=True)
    if not unit_counts.empty:
        print(unit_counts.apply(lambda x: f"{x*100:.2f}%"))
    else:
        print("No units successfully parsed to show distribution.")

    return df_parsed


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