import pandas as pd
import numpy as np
import re

# --- Dosage Parsing Constants ---
DOSAGE_VALUE_COL = 'dosage_value'
DOSAGE_UNIT_COL = 'dosage_unit'

# Updated: Anchored pattern
MOLARITY_RE = re.compile(
    r"^(?P<value>\d+\.?\d*)\s*(?P<prefix>[pnuµm]?)\s*(?P<unit>M)(?:\s|\b)", # Anchored
    re.IGNORECASE
)
MOLARITY_PREFIX_FACTORS = {
    "p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0
}

# Anchored patterns, allowing trailing non-captured space/boundary
DOSAGE_PATTERNS = {
    "ppm":           re.compile(r"^(?P<value>\d+\.?\d*)\s*ppm(?:\s|\b)", re.IGNORECASE),
    "percent":       re.compile(r"^(?P<value>\d+\.?\d*)\s*%(?:\s|\b)", re.IGNORECASE),
    "µg_per_ml":     re.compile(r"^(?P<value>\d+\.?\d*)\s*u?g/ml(?:\s|\b)", re.IGNORECASE),
    "mg_per_ml":     re.compile(r"^(?P<value>\d+\.?\d*)\s*mg/ml(?:\s|\b)", re.IGNORECASE),
    "mg_per_100ml":  re.compile(r"^(?P<value>\d+\.?\d*)\s*mg/100\s*m[l|L](?:\s|\b)", re.IGNORECASE),
    "mg_per_kg":     re.compile(r"^(?P<value>\d+\.?\d*)\s*mg/kg(?:\s|\b)", re.IGNORECASE),
    "mg_per_L":      re.compile(r"^(?P<value>\d+\.?\d*)\s*mg/l(?:\s|\b)", re.IGNORECASE),
    "µl_per_100ml":  re.compile(r"^(?P<value>\d+\.?\d*)\s*u?l/100\s*m[l|L](?:\s|\b)", re.IGNORECASE),
    "g_per_L":       re.compile(r"^(?P<value>\d+\.?\d*)\s*g/l(?:\s|\b)", re.IGNORECASE),
    "µg_per_g":      re.compile(r"^(?P<value>\d+\.?\d*)\s*u?g/g(?:\s|\b)", re.IGNORECASE),
    "iu":            re.compile(r"^(?P<value>\d+\.?\d*)\s*iu(?:\s|\b)", re.IGNORECASE),
    "ng_per_ml":     re.compile(r"^(?P<value>\d+\.?\d*)\s*ng/ml(?:\s|\b)", re.IGNORECASE),
    "mg":            re.compile(r"^(?P<value>\d+\.?\d*)\s*mg(?:\s|\b)", re.IGNORECASE),
    "µmol_per_L":    re.compile(r"^(?P<value>\d+\.?\d*)\s*u?mol/L(?:\s|\b)", re.IGNORECASE),
    "g_per_kg":      re.compile(r"^(?P<value>\d+\.?\d*)\s*g/kg(?:\s|\b)", re.IGNORECASE),
}


def _parse_molarity(s: str) -> tuple[float | None, str | None]:
    """Helper to parse molarity strings. Returns standardized value in M and original unit string."""
    if not isinstance(s, str): return None, None
    try:
        s_cleaned = s.replace("μ", "µ").strip() # Normalize mu
    except AttributeError: return None, None

    m = MOLARITY_RE.search(s_cleaned)
    if not m: return None, None

    try:
        val = float(m.group("value"))
        pfx = m.group("prefix").lower() if m.group("prefix") else ""
        unit_str = pfx + m.group("unit").upper()
        std_value = val * MOLARITY_PREFIX_FACTORS.get(pfx, 1.0)
        return std_value, unit_str
    except (ValueError, AttributeError, TypeError):
        return None, None


def _parse_other_dosage(s: str) -> tuple[float | None, str | None]:
    """Helper to parse non-molarity dosage strings based on defined patterns."""
    if not isinstance(s, str): return None, None
    try:
        # Normalize: replace unicode mu, strip, remove parenthetical footnotes
        txt = s.replace("μ", "µ").strip()
        txt = re.sub(r"\s*\(.*?\)\s*", "", txt).strip()
    except AttributeError:
        return None, "error_normalizing"

    for unit_category, pattern in DOSAGE_PATTERNS.items():
        m = pattern.search(txt)
        if m:
            try:
                return float(m.group("value")), unit_category
            except (ValueError, IndexError):
                 return None, f"error_extracting_{unit_category}"
    return None, None


def parse_dosage_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Parses the dosage column into numeric value and unit category."""
    print("\n--- Comprehensive Dosage Parsing ---")
    if col_name not in df.columns:
        print(f"Warning: '{col_name}' column not found. Skipping parsing.")
        return df

    df_parsed = df.copy()

    def parse_single_dosage(s):
        # Handle NaN/empty/literal 'nan' input
        if pd.isna(s) or (isinstance(s, str) and (not s.strip() or s.strip().lower() == 'nan')):
            return np.nan, "missing"

        # Ensure input is string for regex/parsing helpers
        s_str = str(s)

        # Try molarity parser (returns standardized value in M)
        std_molar_value, _ = _parse_molarity(s_str)
        if std_molar_value is not None:
            return std_molar_value, "molarity"

        # Try other parsers
        other_value, unit_category = _parse_other_dosage(s_str)
        if other_value is not None:
            if unit_category and "error" in unit_category:
                 return np.nan, unit_category
            else:
                 return other_value, unit_category

        # If nothing matched, mark as unknown
        return np.nan, "unknown"

    # Apply the combined parser
    parsed_results = df_parsed[col_name].apply(parse_single_dosage)
    df_parsed[[DOSAGE_VALUE_COL, DOSAGE_UNIT_COL]] = pd.DataFrame(parsed_results.tolist(), index=df_parsed.index)

    # Report Summary Statistics
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

    print("\n--- Dosage Unit Distribution (Successfully Parsed) ---")
    # Exclude 'missing', 'unknown', and errors from value counts of units
    unit_counts = df_parsed.loc[parsed_ok_mask, DOSAGE_UNIT_COL].value_counts(normalize=True)
    if not unit_counts.empty:
        print(unit_counts.apply(lambda x: f"{x*100:.2f}%"))
    else:
        print("No units successfully parsed to show distribution.")

    return df_parsed 