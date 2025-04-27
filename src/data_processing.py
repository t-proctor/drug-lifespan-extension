import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# Configure visualization style
sns.set(style="whitegrid")

# Create directory for plots if it doesn't exist
# Check relative to the script location in drugage/
# plots_dir = 'plots' # Plots are now handled by visualization script
# if not os.path.exists(plots_dir):
#     os.makedirs(plots_dir)

print("--- Loading Data ---")
try:
    # Load from the relative path within the drugage directory
    # df = pd.read_csv('drugage.csv') # OLD PATH
    df = pd.read_csv('../data/raw/drugage.csv') # NEW PATH
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    # Use relative path in error message
    # print("Error: drugage.csv not found in the same directory as the script.") # OLD MSG
    print("Error: ../data/raw/drugage.csv not found. Make sure the raw data exists.") # NEW MSG
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("\n--- Basic Data Info ---")
df.info()

print("\n--- Descriptive Statistics (Initial) ---")
# Use include='all' to get stats for categorical columns too
print(df.describe(include='all'))

print("\n--- Initial Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Cleaning ---")
# Convert relevant columns to numeric, coercing errors
numeric_cols_to_convert = ['avg_lifespan_change_percent', 'max_lifespan_change_percent', 'weight_change_percent']
print(f"Attempting to convert columns to numeric: {numeric_cols_to_convert}")

for col in numeric_cols_to_convert:
    if col in df.columns:
        original_missing = df[col].isnull().sum()
        # Ensure the column is treated as string before potential conversion issues
        df[col] = df[col].astype(str)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        new_missing = df[col].isnull().sum()
        if new_missing > original_missing:
            print(f"  - Column '{col}': Introduced {new_missing - original_missing} new NaNs during conversion (were non-numeric values).")
        elif not pd.api.types.is_numeric_dtype(df[col]):
             print(f"  - Warning: Column '{col}' could not be converted to numeric.")
    else:
        print(f"  - Warning: Column '{col}' not found in DataFrame.")

print("\n--- Missing Values After Numeric Conversion ---")
if all(col in df.columns for col in numeric_cols_to_convert):
    print(df[numeric_cols_to_convert].isnull().sum())
else:
    print("Could not check missing values as some columns were not found during conversion.")

# --- Comprehensive Dosage Parsing ---
print("\n--- Comprehensive Dosage Parsing ---")

# Keep the molarity parser for standardization
molarity_re = re.compile(
    # Capture value, optional prefix, and 'M', case-insensitive
    r"(?P<value>\d+\.?\d*)\s*(?P<prefix>[pnuµm]?)(?P<unit>M)\\b",
    re.IGNORECASE
)
factor = {
    "p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0
}
def parse_molarity(s):
    if not isinstance(s, str): return None, None
    try: s_cleaned = s.replace("μ", "µ").strip()
    except AttributeError: return None, None
    m = molarity_re.search(s_cleaned)
    if not m: return None, None
    try:
        val = float(m.group("value"))
        pfx = m.group("prefix").lower() if m.group("prefix") else ""
        unit_str = pfx + m.group("unit").upper() # e.g., µM, mM, M
        std_value = val * factor.get(pfx, 1.0) # Value in base M
        return std_value, unit_str # Return standardized value and original unit string
    except (ValueError, AttributeError, TypeError): return None, None

# Define other patterns based on Dosage_Pattern_Summary.csv (priority order)
# Using (?P<value>\d+\.?\d*) for consistent float capture
PATTERNS = {
    # pattern_name: regex_compiled
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
    "iu":            re.compile(r"(?P<value>\d+\.?\d*)\s*iu", re.IGNORECASE), # international units
    "ng_per_ml":     re.compile(r"(?P<value>\d+\.?\d*)\s*ng/ml", re.IGNORECASE),
}

def parse_dosage(s):
    # Check for NaN or empty/whitespace string first
    if pd.isna(s) or (isinstance(s, str) and not s.strip()):
        return np.nan, "missing"

    # Normalize: replace unicode mu, strip whitespace, remove parenthetical footnotes
    try:
        txt = s.replace("μ", "µ").strip()
        txt = re.sub(r"\s*\(.*?\)\s*", "", txt).strip() # Remove content within brackets
    except AttributeError: # Should not happen if pd.isna check passed, but belt-and-suspenders
        return np.nan, "error_normalizing"

    # 1. Try molarity parser first (handles unit standardization)
    std_molar_value, _ = parse_molarity(txt) # We only need the value here
    if std_molar_value is not None:
        # Use 'molarity' as the unit category, value is already standardized
        return std_molar_value, "molarity"

    # 2. Iterate through other patterns if molarity didn't match
    for unit_category, pattern in PATTERNS.items():
        m = pattern.search(txt)
        if m:
            try:
                # Return the extracted value and the unit category name
                return float(m.group("value")), unit_category
            except (ValueError, IndexError):
                 # Error during extraction means pattern matched but something went wrong
                 return np.nan, f"error_extracting_{unit_category}"

    # 3. If nothing matched, mark as unknown
    # --- DEBUG PRINT ---
    # print(f"DEBUG: Unmatched format. Original='{s}', Normalized='{txt}'")
    # --- END DEBUG ---
    return np.nan, "unknown"

# Apply the parser
if 'dosage' in df.columns:
    # Ensure dosage is string, apply parser, split results into two new columns
    df['dosage'] = df['dosage'].astype(str)
    parsed_dosage_series = df['dosage'].apply(parse_dosage)
    df[['dosage_value', 'dosage_unit']] = pd.DataFrame(parsed_dosage_series.tolist(), index=df.index)

    # Report Summary Statistics
    print("\n--- Dosage Parsing Results ---")
    total_entries = len(df)
    parsed_ok_mask = df['dosage_value'].notnull()
    parsed_count = parsed_ok_mask.sum()
    missing_explicit_mask = (df['dosage_unit'] == 'missing')
    missing_explicit_count = missing_explicit_mask.sum()
    unknown_mask = (df['dosage_unit'] == 'unknown')
    unknown_count = unknown_mask.sum()
    # Check for any other error states (e.g., 'error_extracting_...')
    error_mask = ~(parsed_ok_mask | missing_explicit_mask | unknown_mask)
    error_count = error_mask.sum()

    print(f"Total entries: {total_entries}")
    print(f" -> Explicitly missing (original NaN): {missing_explicit_count} ({missing_explicit_count/total_entries*100:.2f}%)")
    print(f" -> Successfully parsed (value extracted): {parsed_count} ({parsed_count/total_entries*100:.2f}%)")
    print(f" -> Unmatched/Unknown format: {unknown_count} ({unknown_count/total_entries*100:.2f}%)")
    if error_count > 0:
        print(f" -> Errors during parsing/extraction: {error_count} ({error_count/total_entries*100:.2f}%)")
        # print(df.loc[error_mask, ['dosage', 'dosage_unit']].head()) # Optional: view errors

    print("\n--- Dosage Unit Distribution (Parsed) ---")
    # Exclude 'missing' and 'unknown' and errors from value counts of units
    unit_counts = df.loc[parsed_ok_mask, 'dosage_unit'].value_counts(normalize=True)
    if not unit_counts.empty:
        print(unit_counts.apply(lambda x: f"{x*100:.2f}%"))
    else:
        print("No units successfully parsed to show distribution.")

else:
    print("Warning: 'dosage' column not found. Skipping parsing.")


# --- Imputation and Encoding ---
print("\n--- Imputing Missing Dosage Values & Encoding Units ---")
if 'dosage_value' in df.columns and 'dosage_unit' in df.columns:
    # Impute numerical NaNs with median
    median_dosage = df['dosage_value'].median()
    print(f"Imputing {df['dosage_value'].isnull().sum()} missing dosage values with median ({median_dosage:.4f})")
    df['dosage_value'].fillna(median_dosage, inplace=True)

    # Impute categorical 'dosage_unit' NAs (should only be 'missing' and 'unknown' now)
    # We already handled original NaNs as 'missing'. 'unknown' covers unparsed.
    # No explicit fillna needed here unless errors occurred, but let's check
    if df['dosage_unit'].isnull().any():
         print(f"Warning: Found {df['dosage_unit'].isnull().sum()} unexpected nulls in dosage_unit. Filling with 'unknown'.")
         df['dosage_unit'].fillna('unknown', inplace=True)

    # One-hot encode the dosage unit categories
    print("One-hot encoding dosage units...")
    df = pd.get_dummies(df, columns=['dosage_unit'], prefix='dose', dummy_na=False) # Keep NaNs if any slip through? No.
    print("Columns after one-hot encoding dosage units:", df.columns.tolist())

else:
    print("Skipping imputation/encoding as dosage columns were not generated.")

# --- Save Processed Data ---
print("\n--- Saving Processed Data ---")
# processed_data_path = 'processed_drugage.pkl' # OLD PATH
processed_data_path = '../data/processed/processed_drugage.pkl' # NEW PATH
try:
    df.to_pickle(processed_data_path)
    print(f"Processed data saved successfully to {processed_data_path}")
except Exception as e:
    print(f"Error saving processed data: {e}")


print("\n--- Data Processing Script Finished ---")