# Drug Lifespan Prediction Model

This project aims to predict the percentage change in average lifespan based on data from the DrugAge database and computed chemical descriptors.

It involves:
- Fetching chemical descriptors (SMILES, LogP, TPSA, etc.) for compounds using PubChem.
- Processing and cleaning the combined DrugAge and descriptor data.
- Training a machine learning model (tuned HistGradientBoostingRegressor) to predict lifespan changes.
- Generating SHAP plots and example predictions using a trained model.

## Project Structure

```
.
├── data/
│   ├── raw/              # Original DrugAge data (e.g., drugage.csv)
│   └── processed/        # Processed data files (descriptors, final pkl)
├── models/
│   └── drug_age_w_descriptors_tuned/ # Models trained with imputed descriptors
├── reports/
│   └── figures/          # Generated plots (distributions, comparisons, SHAP)
├── scripts/
│   ├── compute_descriptors.py      # Fetches SMILES & calculates descriptors
│   ├── generate_predictions_table.py # Loads model, generates prediction table
│   ├── generate_shap_plot.py       # Loads model, generates SHAP plot
│   └── visualization.py          # Generates plots from processed data
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Core data processing logic
│   ├── dosage_parser.py    # Dosage string parsing logic
│   └── train_model.py      # Model training and tuning logic
├── .gitignore
├── README.md
└── requirements.txt      # Project dependencies
```

## Usage

The main workflow involves running scripts sequentially:

1.  **Compute Chemical Descriptors:**
    *(Requires internet connection for PubChem lookups)*
    ```bash
    python scripts/compute_descriptors.py
    ```
    This reads `data/raw/drugage.csv`, finds SMILES via PubChem, calculates descriptors using RDKit, and saves the result to `data/processed/drug_descriptors.csv`.

2.  **Process Data:**
    ```bash
    python src/data_processing.py \
        --input_path data/processed/drug_descriptors.csv \
        --output_path data/processed/processed_descriptors_imputed.pkl
    ```
    This script takes the output from step 1 (or the specified `--input_path`), cleans data, parses dosages, imputes missing descriptors, and saves the final processed DataFrame to the specified `--output_path` (defaults to `data/processed/processed_drug_age.pkl` if not provided after recent change, assuming `src/data_processing.py` was modified).

3.  **Train Model:**
    ```bash
    # Trains using imputed descriptors
    python src/train_model.py \
        --input_path data/processed/processed_descriptors_imputed.pkl
    ```
    This script loads processed data, performs hyperparameter tuning (RandomizedSearchCV) for a HistGradientBoostingRegressor, evaluates the best model, and saves the trained pipeline (`best_tuned_model.joblib`) and tuning results (`tuning_cv_results.csv`) to `models/drug_age_w_descriptors_tuned/`.

4.  **Generate Example Predictions (Optional):**
    *(Run after training a model)*
    ```bash
    # Uses default paths for the model trained with imputed descriptors
    python scripts/generate_predictions_table.py \
        --n_samples 15 # Optional: specify number of samples
    ```
    This script loads the saved model pipeline (`models/drug_age_w_descriptors_tuned/best_tuned_model.joblib`) and the corresponding processed data (`data/processed/processed_descriptors_imputed.pkl`), performs the train/test split, makes predictions on the test set, and prints a Markdown table comparing actual vs. predicted values for a random sample.

5.  **Generate SHAP Plot (Optional):**
    *(Run after training a model)*
    ```bash
    # Uses default paths for the model trained with imputed descriptors
    python scripts/generate_shap_plot.py \
        --output_plot_path reports/figures/shap_summary_imputed.png # Optional: specify output path
    ```
    This script loads the saved model pipeline and data, calculates SHAP values for the test set, and saves a SHAP summary plot to the specified path (defaults to `models/drug_age_w_descriptors_tuned/shap_summary_loaded.png`).

6.  **Generate Data Visualizations (Optional):**
    ```bash
    # Assumes data/processed/processed_drug_age.pkl exists
    python scripts/visualization.py
    ```
    This script loads the processed data (hardcoded path currently) and generates various plots (distributions, counts, boxplots, model comparison) saved to `reports/figures/`.

## Results

The performance metrics (R-squared, RMSE) for the trained model are printed to the console during the execution of `src/train_model.py`. Detailed cross-validation results are saved in `models/drug_age_w_descriptors_tuned/tuning_cv_results.csv`.

Example prediction tables and SHAP plots can be generated using the scripts in the `scripts/` directory as described in the Usage section. 