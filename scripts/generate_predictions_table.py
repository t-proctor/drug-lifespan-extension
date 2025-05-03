import pandas as pd
import numpy as np
import joblib
import os
import sys
import argparse
from sklearn.model_selection import train_test_split

# --- Configuration ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = 'avg_lifespan_change_percent'
DEFAULT_MODEL_DIR = 'models/drug_age_w_descriptors'
DEFAULT_MODEL_NAME = 'best_tuned_model.joblib'
DEFAULT_DATA_PATH = 'data/processed/processed_descriptors_imputed.pkl'


def load_data(filepath: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Loads processed data and splits into features (X) and target (y)."""
    print(f"Loading processed data from {filepath}...")
    try:
        df = pd.read_pickle(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        if TARGET_COL not in df.columns:
            print(f"Error: Target column '{TARGET_COL}' not found.", file=sys.stderr)
            return None
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        return X, y
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        return None


def load_model(model_path: str):
    """Loads the trained model pipeline from a joblib file."""
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading joblib file: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained model and generate a table comparing predictions to actual values."
    )
    parser.add_argument(
        '--input_path', type=str, default=DEFAULT_DATA_PATH,
        help=f"Path to the processed input PKL file (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        '--model_path', type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME),
        help=f"Path to the trained model joblib file (default: {os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)})"
    )
    parser.add_argument(
        '--n_samples', type=int, default=20,
        help="Number of samples to include in the output table (default: 20)"
    )
    args = parser.parse_args()

    # Load data
    load_result = load_data(args.input_path)
    if load_result is None:
        sys.exit(1)
    X, y = load_result

    # Load model
    model = load_model(args.model_path)
    if model is None:
        sys.exit(1)

    # Perform the *exact same* train/test split as during training
    print(
        f"Splitting data using test_size={TEST_SIZE} and random_state={RANDOM_STATE}..."
    )
    # We only need the test set here
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Test set shape: {X_test.shape}")

    # Make predictions
    print("Generating predictions on the test set...")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        # Attempting to predict on X_test which might not have been seen by
        # target encoder during training could cause issues if unknown categories exist.
        # The pipeline *should* handle this if configured correctly, but good to note.
        sys.exit(1)


    # --- Generate Example Results Table ---
    print("\n--- Generating Example Results Table ---")

    compound_col = 'compound_name' if 'compound_name' in X_test.columns else None
    if not compound_col:
        print("Warning: 'compound_name' not found. Excluding it from table.")

    results_data = {
        'Actual Lifespan Change (%)': y_test,
        'Predicted Lifespan Change (%)': y_pred
    }
    column_order = ['Actual Lifespan Change (%)', 'Predicted Lifespan Change (%)']

    if compound_col:
        results_data['Compound'] = X_test[compound_col]
        column_order.insert(0, 'Compound') # Put compound first

    results_df = pd.DataFrame(results_data)[column_order]


    # Round the numeric columns for display
    results_df['Actual Lifespan Change (%)'] = results_df['Actual Lifespan Change (%)'].round(2)
    results_df['Predicted Lifespan Change (%)'] = results_df['Predicted Lifespan Change (%)'].round(2)

    # Select a sample
    n_rows_available = len(results_df)
    n_samples_to_show = min(args.n_samples, n_rows_available)

    if n_samples_to_show > 0:
        sample_results = results_df.sample(n=n_samples_to_show, random_state=RANDOM_STATE)
        print(f"Example Predictions vs Actuals (Random Sample - {n_samples_to_show} rows):")
        # Ensure tabulate is installed for to_markdown
        try:
            print(sample_results.to_markdown(index=False))
        except ImportError:
            print("Error: 'tabulate' library not found. Cannot print Markdown table.")
            print("Install it using: pip install tabulate")
            print("\nCSV format fallback:")
            print(sample_results.to_csv(index=False))
        except Exception as e:
            print(f"Error generating table: {e}")
            print("\nCSV format fallback:")
            print(sample_results.to_csv(index=False))
    else:
        print("Not enough results to generate a sample table.")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main() 