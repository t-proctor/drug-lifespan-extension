import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configure visualization style
sns.set(style="whitegrid")

# Define output directory for plots (relative to this script)
plots_dir = '../reports/figures' # NEW PATH
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

print("--- Loading Processed Data ---")
processed_data_path = '../data/processed/processed_drugage.pkl' # NEW PATH
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

# Define column lists needed for plots (based on original script)
lifespan_cols = ['avg_lifespan_change_percent', 'max_lifespan_change_percent']
# Note: Categorical columns might have changed if 'object' types were altered
# during processing, but let's assume these core ones are still relevant.
# The one-hot encoded 'dose_*' columns could also be visualized.
categorical_cols = ['species', 'gender', 'ITP']


# --- Visualizations ---

# 1. Distribution of Lifespan Changes
print("\n--- Generating Lifespan Distribution Plots ---")
plt.figure(figsize=(14, 6))
plot_successful_lifespan = False
for i, col in enumerate(lifespan_cols):
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
        plt.subplot(1, len(lifespan_cols), i + 1)
        # Handle potential infinite values if any survived processing (unlikely but safe)
        data_to_plot = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not data_to_plot.empty:
            sns.histplot(data_to_plot, kde=True, bins=50)
            plot_title = f'Distribution of {col.replace("_percent", " (%)").replace("_", " ").title()}'
            plt.title(plot_title)
            print(f"  - Plotting: {plot_title}")
            plot_successful_lifespan = True
        else:
             print(f"  - Skipping plot for column '{col}' (no valid data after dropping NaN/inf).")
    else:
        print(f"  - Skipping plot for non-numeric, missing, or empty column: {col}")

if plot_successful_lifespan:
    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, 'lifespan_change_distribution.png')
    try:
        plt.savefig(plot_filename)
        print(f"  - Saved plot to {plot_filename}")
    except Exception as e:
        print(f"  - Error saving plot {plot_filename}: {e}")
    plt.close()
else:
    print("  - No valid lifespan columns to plot.")


# 2. Counts of Categorical Features
print("\n--- Generating Categorical Count Plots ---")
for col in categorical_cols:
    if col in df.columns and df[col].notna().any():
        plt.figure(figsize=(10, 8))
        top_n = 20
        # Use the data as loaded (should be appropriate type from pickle)
        col_data = df[col].dropna()
        if col_data.empty:
            print(f"  - Skipping count plot for empty column: {col}")
            plt.close()
            continue

        # Convert to string for consistent counting/plotting if mixed types exist
        col_data = col_data.astype(str)
        value_counts = col_data.value_counts()

        if len(value_counts) > top_n:
            top_categories = value_counts.nlargest(top_n).index
            # Filter the original DataFrame for plotting
            data_to_plot = df[df[col].isin(top_categories)][col].astype(str) # Ensure string type for plotting
            sns.countplot(y=data_to_plot, order=top_categories)
            plt.title(f'Top {top_n} {col.replace("_", " ").title()} Counts')
            print(f"  - Plotting: Top {top_n} {col.title()} Counts (limited due to high cardinality)")
        else:
            order = value_counts.index
            sns.countplot(y=col_data, order=order)
            plt.title(f'{col.replace("_", " ").title()} Counts')
            print(f"  - Plotting: {col.title()} Counts")

        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f'{col}_counts.png')
        try:
            plt.savefig(plot_filename)
            print(f"  - Saved plot to {plot_filename}")
        except Exception as e:
             print(f"  - Error saving plot {plot_filename}: {e}")
        plt.close()
    else:
         print(f"  - Skipping count plot for missing or empty column: {col}")


# 3. Lifespan Change vs. Categorical Features (Boxplots)
print("\n--- Generating Lifespan Change vs Categorical Plots ---")
for cat_col in categorical_cols:
    # Ensure categorical column exists and has data
    if cat_col in df.columns and df[cat_col].notna().any():
        for life_col in lifespan_cols:
             # Ensure lifespan column exists, is numeric, and has data
             if life_col in df.columns and pd.api.types.is_numeric_dtype(df[life_col]) and df[life_col].notna().any():
                plt.figure(figsize=(12, 9))
                top_n = 15
                # Ensure grouping works correctly by handling potential NaN/inf in life_col
                # and ensuring cat_col is suitable for grouping (string usually best)
                df_clean = df[[cat_col, life_col]].copy()
                df_clean[cat_col] = df_clean[cat_col].astype(str) # Ensure string type for reliable grouping
                df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()


                if df_clean[cat_col].nunique() > top_n:
                    # Calculate median on cleaned data
                    median_values = df_clean.groupby(cat_col)[life_col].median()
                    if not median_values.empty:
                        # Get top N categories based on median lifespan change
                        top_categories = median_values.nlargest(top_n).index
                        # Filter the cleaned data for plotting
                        df_filtered = df_clean[df_clean[cat_col].isin(top_categories)]
                        if not df_filtered.empty:
                             # Plot using the filtered data
                             sns.boxplot(data=df_filtered, y=cat_col, x=life_col, order=top_categories)
                             plt.title(f'{life_col.replace("_percent", " (%)").title()} by Top {top_n} {cat_col.title()}')
                             print(f"  - Plotting: {life_col.title()} by Top {top_n} {cat_col.title()} (Boxplot)")
                        else:
                             print(f"  - Skipping boxplot for {life_col} vs {cat_col} (No data after filtering top categories)")
                             plt.close()
                             continue
                    else:
                        print(f"  - Skipping boxplot for {life_col} vs {cat_col} (Could not calculate medians)")
                        plt.close()
                        continue

                else: # Fewer than top_n categories
                    if not df_clean.empty:
                        # Order by median lifespan change
                        order = df_clean.groupby(cat_col)[life_col].median().sort_values(ascending=False).index
                        sns.boxplot(data=df_clean, y=cat_col, x=life_col, order=order)
                        plt.title(f'{life_col.replace("_percent", " (%)").title()} by {cat_col.title()}')
                        print(f"  - Plotting: {life_col.title()} by {cat_col.title()} (Boxplot)")
                    else:
                         print(f"  - Skipping boxplot for {life_col} vs {cat_col} (No valid data)")
                         plt.close()
                         continue

                plt.tight_layout()
                plot_filename = os.path.join(plots_dir, f'{life_col}_vs_{cat_col}_boxplot.png')
                try:
                    plt.savefig(plot_filename)
                    print(f"  - Saved plot to {plot_filename}")
                except Exception as e:
                    print(f"  - Error saving plot {plot_filename}: {e}")
                plt.close()
             else:
                 print(f"  - Skipping boxplot for {life_col} vs {cat_col} (Lifespan column missing, non-numeric, or empty)")
    else:
        print(f"  - Skipping boxplots involving {cat_col} (Categorical column missing or empty)")


print("\n--- Visualization Script Finished ---") 