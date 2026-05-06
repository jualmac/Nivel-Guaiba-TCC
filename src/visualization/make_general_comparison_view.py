"""
Generate prediction comparison plots for the general training mode (Nested CV);

Reads the `models_test_predictions` table from DuckDB, filters by forecasting horizon (step)
and fold, calculates RMSE per model, and saves a stitched comparison chart.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
from sklearn.metrics import root_mean_squared_error

# Add project root to sys.path to allow absolute imports from 'src' when run as a script;
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))  # Add parent dir

from src.db_handler import DBConnection

########################################################################################################################
#                                                                  
# PLOT FUNCTION
#
########################################################################################################################
def make_general_comparison_view(
    step: int = 168,
    fold: Union[int, str] = 'last',
    output_path: str = None,
    db_path: str = "data/nivel_duck.db"
) -> str:
    """
    Build and save a comparison plot of general mode test predictions vs. ground truth;
    
    Args:
        step (int): The forecasting horizon to visualize (e.g., 24, 168, 720).
        fold (int or 'all' or 'last'): Specific fold to show, 'all' to stitch them, or 'last' for the most recent.
        output_path (str): Destination path for the saved plot.
        db_path (str): Path to DuckDB file.
    
    Returns:
        str: Absolute path to the saved plot image;
    """
    # Initialize DB;
    db = DBConnection(path=db_path)

    # Handle 'last' fold detection;
    if fold == 'last':
        max_fold_res = db.run("SELECT MAX(fold) as max_fold FROM models_test_predictions")["result"]
        if not max_fold_res.empty and max_fold_res.iloc[0]['max_fold'] is not None:
            fold = int(max_fold_res.iloc[0]['max_fold'])
        else:
            raise ValueError("Could not determine last fold; table might be empty.")

    if output_path is None:
        fold_str = f"fold_{fold}" if fold != 'all' else "all_folds"
        output_path = f"data/general_comparison_{step}h_{fold_str}.png"

    # Ensure output directory exists;
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load predictions table;
    query = f"SELECT * FROM models_test_predictions WHERE step = {step}"
    if fold != 'all':
        query += f" AND fold = {fold}"
    
    df = db.run(query)["result"]
    
    if df.empty:
        raise ValueError(f"No data found in models_test_predictions for step={step} and fold={fold}")

    # Ensure chronological order;
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'model_name'])

    # Pivot to have one column per model;
    # Columns: date, model1, model2, ..., y_true
    pivot_df = df.pivot_table(index='date', columns='model_name', values='y_pred')
    y_true = df.groupby('date')['y_true'].first()
    pivot_df['y_true'] = y_true
    pivot_df = pivot_df.reset_index()

    # Identify model columns;
    exclude_cols = ["date", "y_true"]
    model_cols = [c for c in pivot_df.columns if c not in exclude_cols]

    # Calculate RMSE for each model;
    model_labels = {}
    for model in model_cols:
        # Drop NaNs for RMSE calculation (in case some models have missing predictions);
        valid_mask = pivot_df[model].notna() & pivot_df['y_true'].notna()
        if valid_mask.any():
            rmse_val = root_mean_squared_error(pivot_df.loc[valid_mask, 'y_true'], pivot_df.loc[valid_mask, model])
            model_labels[model] = f"{model} (RMSE: {rmse_val:.2f} cm)"
        else:
            model_labels[model] = model

    # Melt for plotting;
    plot_df = pivot_df.melt(
        id_vars=["date", "y_true"],
        value_vars=model_cols,
        var_name="model",
        value_name="y_pred"
    )
    # Apply labels with RMSE;
    plot_df['model_label'] = plot_df['model'].map(model_labels)

    # Plot;
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sns.lineplot(data=plot_df, x="date", y="y_pred", hue="model_label", ax=ax, alpha=0.8)
    sns.lineplot(data=pivot_df, x="date", y="y_true", color="black", linewidth=2, label="Observado (y_true)", ax=ax)
    
    title_suffix = f"Fold {fold}" if fold != 'all' else "Todos os Folds (Stitched)"
    ax.set_title(f"General Mode: Comparação de Modelos (Horizonte: {step}h) - {title_suffix}", fontsize=14)
    ax.set_xlabel("Data")
    ax.set_ylabel("Nível (cm)")
    ax.legend(title="Modelos", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate general mode comparison plots")
    parser.add_argument('--step', type=int, default=360, help='Forecasting horizon (hours)')
    parser.add_argument('--fold', type=str, default='all', help='Fold number, "all", or "last" (default)')
    
    args = parser.parse_args()
    
    # Convert fold to int if it's numeric;
    fold_val = args.fold
    if fold_val.isdigit():
        fold_val = int(fold_val)
        
    try:
        saved_path = make_general_comparison_view(step=args.step, fold=fold_val)
        print(f"Saved general comparison plot to: {saved_path}")
    except Exception as e:
        print(f"Error: {e}")
