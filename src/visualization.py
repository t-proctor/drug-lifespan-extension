import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configure visualization style
sns.set_theme(style="whitegrid")
sns.set_palette("Greens")

# Define output directory for plots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
plots_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# --- Helper Functions ---


def _format_title(col_name, prefix="Distribution of", suffix=""):
    """Formats a column name into a plot title."""
    formatted_name = col_name.replace(
        "_percent", " (%)"
    ).replace("_", " ").title()
    return f"{prefix} {formatted_name}{suffix}"


def _save_plot(plt_obj, filename, directory):
    """Saves the current matplotlib plot to a file."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"  - Created directory during save: {directory}")
        except Exception as e:
            print(f"  - Error creating directory {directory}: {e}")
            return
    full_path = os.path.join(directory, filename)
    try:
        plt_obj.savefig(full_path)
        print(f"  - Saved plot to {full_path}")
    except Exception as e:
        print(f"  - Error saving plot {full_path}: {e}")
    finally:
        plt_obj.close()


def plot_metric_comparison(metric_name, data, filename, directory):
    """Generates and saves a bar plot comparing model performance metrics."""
    plt.figure(figsize=(10, 6))
    model_names = list(data.keys())
    metric_values = list(data.values())

    bars = sns.barplot(x=model_names, y=metric_values, palette="Greens")

    # Add metric values on top of bars
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}',
            va='bottom', ha='center'
        )

    # Use RMSE or R-squared in the title/labels as appropriate
    ylabel = metric_name
    title = f'Model Comparison: {metric_name}'
    if metric_name == "RMSE":
        ylabel = "RMSE (Test)"
        title = "Model Comparison: Root Mean Squared Error (Test)"
    elif metric_name == "R2":
        ylabel = "R² (Test)"
        title = "Model Comparison: R-squared (Test)"

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    _save_plot(plt, filename, directory)


# --- Data Loading ---
print("--- Loading Processed Data ---")
processed_data_path = '../data/processed/processed_drugage.pkl'
try:
    df = pd.read_pickle(processed_data_path)
    print(f"Successfully loaded processed data from {processed_data_path}")
    print(f"Data shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Processed data file not found at {processed_data_path}")
    print("Please run the data processing script first (e.g., eda_script.py).")
    exit()
except Exception as e:
    print(f"Error loading processed data: {e}")
    exit()

# Define column lists needed for plots
lifespan_cols = ['avg_lifespan_change_percent', 'max_lifespan_change_percent']
categorical_cols = ['species', 'gender', 'ITP']


# --- Visualizations ---

# 1. Distribution of Lifespan Changes
print("\n--- Generating Lifespan Distribution Plots ---")
plt.figure(figsize=(14, 6))
plot_successful_lifespan = False
subplot_index = 1
num_lifespan_cols = len(lifespan_cols)

for col in lifespan_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        # Handle potential infinite values and NaNs
        data_to_plot = df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if not data_to_plot.empty:
            plt.subplot(1, num_lifespan_cols, subplot_index)
            sns.histplot(data_to_plot, kde=True, bins=50)
            plot_title = _format_title(col, prefix="Distribution of")
            plt.title(plot_title)
            print(f"  - Plotting: {plot_title}")
            plot_successful_lifespan = True
            subplot_index += 1
        else:
            print(
                f"  - Skipping plot for column '{col}' "
                f"(no valid data after dropping NaN/inf)."
            )
    else:
        print(f"  - Skipping plot for non-numeric or missing column: {col}")


if plot_successful_lifespan:
    plt.tight_layout()
    plot_filename = 'lifespan_change_distribution.png'
    _save_plot(plt, plot_filename, plots_dir)
else:
    print("  - No valid lifespan columns to plot.")
    plt.close()


# 2. Counts of Categorical Features
print("\n--- Generating Categorical Count Plots ---")
for col in categorical_cols:
    if col in df.columns and df[col].notna().any():
        col_data = df[col].dropna()
        if col_data.empty:
            print(f"  - Skipping count plot for empty column: {col}")
            continue

        # Convert to string for consistent counting/plotting
        col_data = col_data.astype(str)
        value_counts = col_data.value_counts()

        plt.figure(figsize=(10, 8))
        top_n = 20
        plot_title = _format_title(col, prefix="")

        if len(value_counts) > top_n:
            top_categories = value_counts.nlargest(top_n).index
            data_to_plot = col_data[col_data.isin(top_categories)]
            sns.countplot(y=data_to_plot, order=top_categories)
            plot_title = _format_title(
                col, prefix=f"Top {top_n}", suffix=" Counts"
            )
            print(
                f"  - Plotting: {plot_title} "
                "(limited due to high cardinality)"
            )
        else:
            order = value_counts.index
            sns.countplot(y=col_data, order=order)
            plot_title = _format_title(col, prefix="", suffix=" Counts")
            print(f"  - Plotting: {plot_title}")

        plt.title(plot_title)
        plt.tight_layout()
        plot_filename = f'{col}_counts.png'
        _save_plot(plt, plot_filename, plots_dir)

    else:
        print(f"  - Skipping count plot for missing or empty column: {col}")


# 3. Lifespan Change vs. Categorical Features (Boxplots)
print("\n--- Generating Lifespan Change vs Categorical Plots ---")
for cat_col in categorical_cols:
    if cat_col not in df.columns or df[cat_col].isnull().all():
        print(
            f"  - Skipping boxplots involving {cat_col} "
            "(Categorical column missing or empty)"
        )
        continue

    for life_col in lifespan_cols:
        if (life_col not in df.columns or
                not pd.api.types.is_numeric_dtype(df[life_col]) or
                df[life_col].isnull().all()):
            print(
                f"  - Skipping boxplot for {life_col} vs {cat_col} "
                "(Lifespan column missing, non-numeric, or empty)"
            )
            continue

        # Prepare data
        df_clean = df[[cat_col, life_col]].copy()
        df_clean[cat_col] = df_clean[cat_col].astype(str)
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

        if df_clean.empty:
            print(
                f"  - Skipping boxplot for {life_col} vs {cat_col} "
                "(No valid data after cleaning)"
            )
            continue

        plt.figure(figsize=(12, 9))
        top_n = 15
        order = None
        plot_title_prefix = _format_title(life_col, prefix="", suffix="")
        plot_title_suffix = f" by {cat_col.title()}"
        print_prefix = f"  - Plotting: {plot_title_prefix} by"

        # Determine categories and order
        unique_categories = df_clean[cat_col].nunique()
        median_values = df_clean.groupby(cat_col)[life_col].median()

        if median_values.empty:
            print(
                f"  - Skipping boxplot for {life_col} vs {cat_col} "
                "(Could not calculate medians for ordering/filtering)"
            )
            plt.close()
            continue

        if unique_categories > top_n:
            top_categories = median_values.nlargest(top_n).index
            df_plot = df_clean[df_clean[cat_col].isin(top_categories)]
            order = top_categories
            plot_title_suffix = f" by Top {top_n} {cat_col.title()}"
            print_prefix = f"  - Plotting: {plot_title_prefix} by Top {top_n}"
            print(f"{print_prefix} {cat_col.title()} (Boxplot)")

        else:
            order = median_values.sort_values(ascending=False).index
            df_plot = df_clean
            print(f"{print_prefix} {cat_col.title()} (Boxplot)")

        # Generate the plot
        if not df_plot.empty:
            sns.boxplot(data=df_plot, y=cat_col, x=life_col, order=order)
            plt.title(f"{plot_title_prefix}{plot_title_suffix}")
            plt.tight_layout()
            plot_filename = f'{life_col}_vs_{cat_col}_boxplot.png'
            _save_plot(plt, plot_filename, plots_dir)
        else:
            print(
                f"  - Skipping boxplot for {life_col} vs {cat_col} "
                "(No data after filtering top categories)"
            )
            plt.close()


# 4. Model Performance Comparison Plots
print("\n--- Generating Model Performance Comparison Plots ---")

# Hardcoded model metrics for plotting
model_metrics = {
    'DrugAge Only': {'R2': 0.1971, 'RMSE': 18.3527},
    'DrugAge Only (Tuned)': {'R2': 0.1957, 'RMSE': 18.3684},
    'DrugAge + Chem': {'R2': 0.2143, 'RMSE': 18.1547},
    'DrugAge + Chem (Tuned)': {'R2': 0.2286, 'RMSE': 17.9890}
}

# Prepare data for plotting
r_squared_data = {name: metrics['R2']
                  for name, metrics in model_metrics.items()}
mse_data = {name: metrics['RMSE']**2 for name,
            metrics in model_metrics.items()}


# Plot R-squared Comparison
plot_metric_comparison(
    metric_name="R2",
    data=r_squared_data,
    filename="model_comparison_r_squared.png",
    directory=plots_dir
)

# Plot MSE Comparison
plot_metric_comparison(
    metric_name="MSE",
    data=mse_data,
    filename="model_comparison_mse.png",
    directory=plots_dir
)


print("\n--- Visualization Script Finished ---")
