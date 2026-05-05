"""
Generate prediction comparison plots from stored model outputs;

Reads the `models_case_predictions` table from DuckDB, reshapes predictions for
line plotting, overlays ground truth, and saves the chart under ./data/;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import numpy as np
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.db_handler import DBConnection
from src.metrics import kge

########################################################################################################################
#                                                                  
# PLOT FUNCTION
#
########################################################################################################################
def make_prediction_view(
    output_path: str = "data/predictions_comparison.png",
    db_path: str = "data/nivel_duck.db",
    step: int = 24
) -> str:
    """
    Build and save a comparison plot of model predictions vs. ground truth;
    
    Args:
        output_path (str): Destination path for the saved plot (default: data/predictions_comparison.png);
        db_path (str): Path to DuckDB file (default: data/nivel_duck.db);
        step (int): Forecasting horizon to plot when the table contains multiple horizons;
    
    Returns:
        str: Absolute path to the saved plot image;
    """
    # Ensure output directory exists to avoid IO errors at save time;
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load predictions table from DuckDB; supports legacy single-horizon columns and horizon-suffixed columns;
    db = DBConnection(path=db_path)
    df = db.run("SELECT * FROM models_case_predictions ORDER BY date")["result"]
    if df.empty:
        raise ValueError("models_case_predictions is empty; rerun backend to generate predictions")
    
    # df.drop(columns=['LSTM', 'SARIMA'], inplace=True)

    df = df.loc[df['date'] > '2024-05-01']
    # # df = df.loc[df['date'] < '2025-08-01']

    # df['LIGHTGBM'] = df['LIGHTGBM'] + 50
    # df['XGBOOST'] = df['XGBOOST'] + 50

    # Normalize datetime column for consistent plotting; fallback to original if conversion fails;
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass  # Keep as-is if coercion fails;

    step_suffix = f"_{step}h"
    horizon_cols = [
        col for col in df.columns
        if col.endswith(step_suffix) and not col.startswith("y_true_")
    ]
    y_true_horizon_col = f"y_true_{step}h"

    if horizon_cols:
        if y_true_horizon_col not in df.columns:
            raise ValueError(f"No observed target column found for step={step}.")

        # Select one forecast horizon explicitly because each horizon has its own observed target;
        rename_cols = {
            col: col.removesuffix(step_suffix)
            for col in horizon_cols
        }
        rename_cols[y_true_horizon_col] = "y_true"
        df = df[["date", y_true_horizon_col] + horizon_cols].rename(columns=rename_cols)
    elif "y_true" not in df.columns:
        raise ValueError(f"No y_true column found for step={step}; rerun backend to generate case predictions.")
    
    # Identify model prediction columns by excluding date and y_true;
    exclude_cols: List[str] = ["date", "y_true"]
    model_cols = [c for c in df.columns if c not in exclude_cols]
    if not model_cols:
        raise ValueError("No model prediction columns found in models_case_predictions")

    for model in ["XGBOOST", "LIGHTGBM"]:
        if model in model_cols:
            print(kge(
                y_true=df['y_true'].to_numpy(dtype=float),
                y_pred=df[model].to_numpy(dtype=float)
            ))
    
    # Melt predictions to long format for seaborn hue handling; keeps y_true separate for emphasis;
    plot_df = df.melt(
        id_vars=["date", "y_true"],
        value_vars=model_cols,
        var_name="model",
        value_name="y_pred"
    )

    # Plot predictions per model with ground truth overlay; black line highlights truth trajectory;
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x="date", y="y_pred", hue="model", ax=ax, alpha=0.85)
    sns.lineplot(data=df, x="date", y="y_true", color="black", linewidth=2.2, label="y_true", ax=ax)
    
    ax.set_title(f"Comparação da predição dos modelos e observado ({step}h)", fontsize=13)
    ax.set_xlabel("Data")
    ax.set_ylabel("Nível (cm)")
    ax.legend()
    fig.autofmt_xdate()
    
    # Persist figure and clean up pyplot state to avoid leaks in repeated calls;
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    saved_path = make_prediction_view()
    print(f"Saved plot to {saved_path}")
