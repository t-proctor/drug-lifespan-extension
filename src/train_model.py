import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import joblib
import os
import sys
import argparse
from scipy.stats import uniform, randint

# --- Configuration ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5  # Number of folds for cross-validation within tuning
N_TUNING_ITER = 50  # Number of parameter settings to sample

# Define target and feature columns based on the processed data
TARGET_COL = 'avg_lifespan_change_percent'
DOSAGE_VALUE_COL = 'dosage_value'
DESCRIPTOR_COLS = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']


def main():
    parser = argparse.ArgumentParser(
        description="Train and tune a lifespan prediction model."
    )
    parser.add_argument(
        '--input_path', type=str, required=True,
        help="Path to the processed input PKL file."
    )
    parser.add_argument(
        '--data_source', type=str,
        choices=['with_descriptors', 'no_descriptors'],
        default='with_descriptors',
        help="Specify data: 'with_descriptors' or 'no_descriptors' (default)."
    )
    args = parser.parse_args()

    # Define output directory based on data source
    if args.data_source == 'with_descriptors':
        MODEL_OUTPUT_DIR = 'models/drug_age_w_descriptors_tuned'
    elif args.data_source == 'no_descriptors':
        MODEL_OUTPUT_DIR = 'models/drug_age_only_tuned'
    else:
        raise ValueError(f"Invalid data_source: {args.data_source}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("--- Starting Model Training & Tuning ---")
    print(f"Input data file: {args.input_path}")
    print(f"Data source type: {args.data_source}")
    print(f"Output directory: {MODEL_OUTPUT_DIR}")
    print(f"Tuning iterations: {N_TUNING_ITER}, CV folds: {N_CV_FOLDS}")

    # --- Data Loading and Preparation ---
    print(f"Loading processed data from {args.input_path}...")
    try:
        df = pd.read_pickle(args.input_path)
    except FileNotFoundError:
        print(
            f"Error: Processed data file not found at {args.input_path}. "
            "Run data processing first.", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Data loaded successfully. Shape: {df.shape}")

    # --- Conditionally Drop Descriptors ---
    if args.data_source == 'no_descriptors':
        cols_to_drop = [col for col in DESCRIPTOR_COLS if col in df.columns]
        if cols_to_drop:
            print(
                f"Data source is '{args.data_source}'. Dropping descriptor "
                f"columns: {cols_to_drop}..."
            )
            df = df.drop(columns=cols_to_drop)
            print(f"DataFrame shape after dropping descriptors: {df.shape}")
        else:
            print(
                f"Data source is '{args.data_source}', but no descriptor "
                "columns found to drop."
            )

    if TARGET_COL not in df.columns:
        print(
            f"Error: Target column '{TARGET_COL}' not found in "
            "the input data.",
            file=sys.stderr
        )
        sys.exit(1)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Dynamically identify feature types
    all_cols = X.columns.tolist()
    categorical_features_present = [
        col for col in ['species', 'strain', 'gender', 'ITP']
        if col in all_cols
    ]
    high_cardinality_features_present = [
        'compound_name'
    ] if 'compound_name' in all_cols else []
    low_cardinality_features_present = [
        f for f in categorical_features_present
        if f not in high_cardinality_features_present
    ]
    dosage_value_col_present = [
        DOSAGE_VALUE_COL
    ] if DOSAGE_VALUE_COL in all_cols else []

    print(
        f"Splitting data into train/test sets (test_size={TEST_SIZE}, "
        f"random_state={RANDOM_STATE})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- Preprocessing Pipeline Definition ---
    target_encoder = ce.TargetEncoder(
        cols=high_cardinality_features_present,
        handle_missing='value', handle_unknown='value'
    )
    one_hot_encoder = OneHotEncoder(
        handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    transformer_list = []
    if high_cardinality_features_present:
        transformer_list.append((
            'target_enc', target_encoder, high_cardinality_features_present
        ))
    if low_cardinality_features_present:
        transformer_list.append((
            'one_hot', one_hot_encoder, low_cardinality_features_present
        ))
    if dosage_value_col_present:
        transformer_list.append(('scaler', scaler, dosage_value_col_present))

    preprocessor = ColumnTransformer(
        transformers=transformer_list,
        remainder='passthrough'
    )

    # --- Model Definition ---
    base_hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    # --- Pipeline Definition ---
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', base_hgb)
    ])

    # --- Hyperparameter Tuning Setup ---
    print("\n--- Setting up Hyperparameter Tuning (RandomizedSearchCV) ---")
    # Define parameter distributions to sample from
    param_dist = {
        'regressor__learning_rate': uniform(0.01, 0.3),  # Range [0.01, 0.31)
        'regressor__max_iter': randint(100, 500),  # Trees [100, 499]
        'regressor__max_leaf_nodes': randint(15, 60),  # Max nodes [15, 59]
        'regressor__max_depth': [None] + list(randint(3, 15).rvs(5)),  # Depth
        'regressor__min_samples_leaf': randint(10, 50),  # Min samples [10, 49]
        'regressor__l2_regularization': uniform(0, 1.0),  # L2 [0, 1.0)
    }

    # Define the cross-validation strategy for tuning
    inner_cv = KFold(n_splits=N_CV_FOLDS, shuffle=True,
                     random_state=RANDOM_STATE)

    # Define the scoring metric
    scoring = 'neg_root_mean_squared_error'

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_TUNING_ITER,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=-1,  # Use all available CPU cores
        random_state=RANDOM_STATE,
        verbose=1  # Show progress
    )

    # --- Run Tuning ---
    print(
        f"Running RandomizedSearchCV with {N_TUNING_ITER} iterations and "
        f"{N_CV_FOLDS}-fold CV..."
    )
    random_search.fit(X_train, y_train)

    print("\n--- Tuning Complete ---")
    print("Best parameters found:")
    print(random_search.best_params_)
    print(f"Best CV score ({scoring}): {random_search.best_score_:.4f}")

    # Get the best pipeline
    best_model = random_search.best_estimator_

    # --- Evaluation on Hold-Out Test Set ---
    print("\n--- Evaluating Best Model on Hold-Out Test Set ---")
    y_pred_best = best_model.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    print(f"***** Tuned HGB R-squared (Test): {r2_best:.4f} *****")
    print(f"***** Tuned HGB RMSE (Test): {rmse_best:.4f} *****")

    # --- Save Best Model ---
    model_filename = "best_tuned_model.joblib"
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
    print(f"\nSaving best tuned model pipeline to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved.")

    # --- Save Tuning Results ---
    try:
        cv_results_df = pd.DataFrame(random_search.cv_results_)
        cv_results_df = cv_results_df.sort_values(
            by='rank_test_score', ascending=True
        )
        cv_results_path = os.path.join(
            MODEL_OUTPUT_DIR, 'tuning_cv_results.csv'
        )
        cv_results_df.to_csv(cv_results_path, index=False)
        print(f"Tuning cross-validation results saved to {cv_results_path}")
    except Exception as e:
        print(f"Warning: Could not save tuning results: {e}")

    print("\n--- Model Training & Tuning Script Finished ---")


if __name__ == "__main__":
    main()
