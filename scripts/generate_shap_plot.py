import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
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
DEFAULT_PLOT_NAME = 'shap_summary_loaded.png'


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
        description="Load a trained model and generate a SHAP summary plot."
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
        '--output_plot_path', type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_PLOT_NAME),
        help=f"Path to save the SHAP summary plot (default: {os.path.join(DEFAULT_MODEL_DIR, DEFAULT_PLOT_NAME)})"
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
    _, X_test, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Test set shape: {X_test.shape}")

    # --- SHAP Analysis --- 
    print("\n--- Calculating SHAP Values --- ")
    try:
        print("Transforming test data using model's preprocessor...")
        # Access the preprocessor step from the loaded pipeline
        preprocessor = model.named_steps['preprocess']
        X_test_processed = preprocessor.transform(X_test)
        print(f"Processed test data shape for SHAP: {X_test_processed.shape}")

        # Get feature names after preprocessing
        feature_names_out = preprocessor.get_feature_names_out()
        print(f"Retrieved {len(feature_names_out)} feature names after preprocessing.")

        if (feature_names_out is not None and
                len(feature_names_out) == X_test_processed.shape[1]):
             X_test_processed_df = pd.DataFrame(
                 X_test_processed, columns=feature_names_out, index=X_test.index
             )
        else:
             print("Warning: Feature names mismatch or unavailable. Using default indices.")
             X_test_processed_df = pd.DataFrame(X_test_processed, index=X_test.index)

    except Exception as e:
        print(f"Error during preprocessing or feature name retrieval: {e}", file=sys.stderr)
        sys.exit(1)

    # Use TreeExplainer with the regressor from the loaded pipeline
    try:
        print("Creating SHAP explainer...")
        regressor = model.named_steps['regressor']
        explainer = shap.TreeExplainer(regressor)
        print("Calculating SHAP values for the test set...")
        shap_values = explainer.shap_values(X_test_processed)

        print("Generating SHAP summary plot...")
        shap.summary_plot(shap_values, X_test_processed_df, show=False)

        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_plot_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        plt.savefig(args.output_plot_path, bbox_inches='tight')
        print(f"SHAP summary plot saved to {args.output_plot_path}")
        plt.close()

    except Exception as e:
        print(f"Error during SHAP calculation or plotting: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main() 