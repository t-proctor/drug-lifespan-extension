import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce # For target encoding
import joblib # For saving models
import os
import sys
import argparse
from scipy.stats import uniform, randint # Added for parameter distributions

# --- Configuration ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5 # Number of folds for cross-validation within tuning
N_TUNING_ITER = 50 # Number of parameter settings to sample in RandomizedSearch

# Define target and feature columns based on the processed data
TARGET_COL = 'avg_lifespan_change_percent'
DOSAGE_VALUE_COL = 'dosage_value'
DESCRIPTOR_COLS = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']

def main():
    parser = argparse.ArgumentParser(description="Train and tune a lifespan prediction model.")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the processed input PKL file (e.g., with imputed descriptors).")
    parser.add_argument('--data_source', type=str, choices=['with_descriptors', 'no_descriptors'], default='with_descriptors',
                        help="Specify data to use: 'with_descriptors' (use chem descriptors) or 'no_descriptors' (drop descriptors).")
    args = parser.parse_args()

    # Define output directory based on data source
    if args.data_source == 'with_descriptors':
        MODEL_OUTPUT_DIR = 'models/drug_age_w_descriptors'
    elif args.data_source == 'no_descriptors':
        MODEL_OUTPUT_DIR = 'models/drug_age_only'
    else:
        raise ValueError(f"Invalid data_source: {args.data_source}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) # Ensure output dir exists

    print(f"--- Starting Model Training & Tuning --- ")
    print(f"Input data file: {args.input_path}")
    print(f"Data source type: {args.data_source}")
    print(f"Output directory: {MODEL_OUTPUT_DIR}")
    print(f"Tuning iterations: {N_TUNING_ITER}, CV folds: {N_CV_FOLDS}")

    print(f"Loading processed data from {args.input_path}...")
    try:
        df = pd.read_pickle(args.input_path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {args.input_path}. Run data processing first.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Data loaded successfully. Shape: {df.shape}")

    # --- Conditionally Drop Descriptors ---
    if args.data_source == 'no_descriptors':
        cols_to_drop = [col for col in DESCRIPTOR_COLS if col in df.columns]
        if cols_to_drop:
            print(f"Data source is '{args.data_source}'. Dropping descriptor columns: {cols_to_drop}...")
            df = df.drop(columns=cols_to_drop)
            print(f"DataFrame shape after dropping descriptors: {df.shape}")
        else:
            print(f"Data source is '{args.data_source}', but no descriptor columns found to drop.")

    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in the input data.", file=sys.stderr)
        sys.exit(1)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Dynamically identify feature types
    all_cols = X.columns.tolist()
    categorical_features_present = [col for col in ['species', 'strain', 'gender', 'ITP'] if col in all_cols]
    high_cardinality_features_present = ['compound_name'] if 'compound_name' in all_cols else []
    low_cardinality_features_present = [f for f in categorical_features_present if f not in high_cardinality_features_present]
    descriptor_cols_present = [col for col in DESCRIPTOR_COLS if col in all_cols]
    dosage_value_col_present = [DOSAGE_VALUE_COL] if DOSAGE_VALUE_COL in all_cols else []
    dose_cols_present = [col for col in all_cols if col.startswith('dose_')]
    passthrough_cols = descriptor_cols_present + dose_cols_present


    print(f"Splitting data into train/test sets (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- Preprocessing Pipeline Definition ---
    target_encoder = ce.TargetEncoder(cols=high_cardinality_features_present, handle_missing='value', handle_unknown='value')
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    transformer_list = []
    if high_cardinality_features_present:
        transformer_list.append(('target_enc', target_encoder, high_cardinality_features_present))
    if low_cardinality_features_present:
        transformer_list.append(('one_hot', one_hot_encoder, low_cardinality_features_present))
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
        'regressor__learning_rate': uniform(0.01, 0.3), # Learning rate between 0.01 and 0.31
        'regressor__max_iter': randint(100, 500), # Number of boosting iterations (trees)
        'regressor__max_leaf_nodes': randint(15, 60), # Max nodes per tree (controls complexity)
        'regressor__max_depth': [None] + list(randint(3, 15).rvs(5)), # Max depth (None means unlimited until max_leaf_nodes) - sample a few depths
        'regressor__min_samples_leaf': randint(10, 50), # Min samples required at a leaf node
        'regressor__l2_regularization': uniform(0, 1.0), # L2 regularization strength
    }

    # Define the cross-validation strategy for tuning
    inner_cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Define the scoring metric (maximize negative RMSE -> minimize RMSE)
    scoring = 'neg_root_mean_squared_error'

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_TUNING_ITER,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    # --- Run Tuning ---
    print(f"Running RandomizedSearchCV with {N_TUNING_ITER} iterations and {N_CV_FOLDS}-fold CV...")
    random_search.fit(X_train, y_train)

    print("\n--- Tuning Complete ---")
    print(f"Best parameters found:")
    print(random_search.best_params_)
    print(f"Best cross-validation score ({scoring}): {random_search.best_score_:.4f}") # Note: This is negative RMSE

    # Get the best pipeline (already refitted on the whole training set)
    best_model = random_search.best_estimator_

    # --- Evaluation on Hold-Out Test Set --- 
    print("\n--- Evaluating Best Model on Hold-Out Test Set ---")
    y_pred_best = best_model.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    print(f"***** Tuned HGB R-squared (Test): {r2_best:.4f} *****")
    print(f"***** Tuned HGB RMSE (Test): {rmse_best:.4f} *****")

    # --- SHAP Analysis (Using the Best Model) --- 
    print("\n--- Calculating SHAP Values for Best Model --- ")
    # Apply preprocessor from the best pipeline
    try:
        print("Transforming test data using best model's preprocessor...")
        # The preprocessor is already fitted within the best_model pipeline after random_search.fit
        X_test_processed = best_model.named_steps['preprocess'].transform(X_test)
        print(f"Processed test data shape for SHAP: {X_test_processed.shape}")

        # Get feature names
        feature_names_out = best_model.named_steps['preprocess'].get_feature_names_out()
        print(f"Successfully retrieved {len(feature_names_out)} feature names after preprocessing.")
        
        if feature_names_out is not None and len(feature_names_out) == X_test_processed.shape[1]:
             X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index) # Keep index
        else:
             print("Warning: Processed feature names mismatch or unavailable. SHAP plot may lack names.")
             X_test_processed_df = pd.DataFrame(X_test_processed, index=X_test.index) # Use numpy array

    except Exception as e:
        print(f"Error during preprocessing or feature name retrieval for SHAP: {e}", file=sys.stderr)
        sys.exit(1)

    # Use TreeExplainer with the regressor from the best pipeline
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(best_model.named_steps['regressor'])
    print("Calculating SHAP values for the test set...")
    shap_values = explainer.shap_values(X_test_processed)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_test_processed_df, show=False) 
    
    shap_plot_filename = f"shap_summary_tuned.png"
    shap_plot_path = os.path.join(MODEL_OUTPUT_DIR, shap_plot_filename)
    plt.savefig(shap_plot_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to {shap_plot_path}")
    plt.close()

    # --- Save Best Model --- 
    model_filename = f"best_tuned_model.joblib"
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
    print(f"\nSaving best tuned model pipeline to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved.")

    # --- Save Tuning Results ---
    try:
        cv_results_df = pd.DataFrame(random_search.cv_results_)
        # Sort results for easier analysis
        cv_results_df = cv_results_df.sort_values(by='rank_test_score', ascending=True)
        cv_results_path = os.path.join(MODEL_OUTPUT_DIR, 'tuning_cv_results.csv')
        cv_results_df.to_csv(cv_results_path, index=False)
        print(f"Tuning cross-validation results saved to {cv_results_path}")
    except Exception as e:
        print(f"Warning: Could not save tuning results: {e}")


    print("\n--- Model Training & Tuning Script Finished ---")

if __name__ == "__main__":
    main() 