"""
This file includes alternative metrics to be used in statistical analysis. These metrics are not allways includes in 
packages like scikit learn. However, some of them may depend on these packages on someway;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import logging
import numpy as np
from permetrics.regression import RegressionMetric
from src.util import configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# FUNCTIONS
#
########################################################################################################################

def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Normalized Nash-Sutcliffe Efficiency (NSE);
    """
    nse = 1-(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return 1/(2-nse)


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Kling Gupta Efficiency (KGE);
    """
    # Ensure numpy arrays for permetrics input;
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    evaluator = RegressionMetric(y_true=y_true_arr, y_pred=y_pred_arr)
    return evaluator.kling_gupta_efficiency()


def evaluate_multi_horizon(y_true: np.ndarray, y_pred: np.ndarray, steps: list) -> dict:
    """
    Evaluate metrics for different forecasting horizons.
    
    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        steps (list): List of integers representing horizons (e.g. [24, 72, 168]).
    
    Returns:
        dict: Metrics computed for each step interval. For a step N, metrics are evaluated on the slice [0:N].
              If prediction length is shorter than N, evaluates up to the available length.
    """
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    results = {}
    
    for step in sorted(steps):
        # We slice up to the step horizon. If array is shorter, it uses whatever is available.
        y_true_slice = y_true[:step]
        y_pred_slice = y_pred[:step]
        
        if len(y_true_slice) == 0:
            continue
            
        rmse = root_mean_squared_error(y_true_slice, y_pred_slice)
        mae = mean_absolute_error(y_true_slice, y_pred_slice)
        try:
            nse_val = nse(y_true_slice, y_pred_slice)
        except Exception:
            nse_val = np.nan
            
        try:
            r2 = r2_score(y_true_slice, y_pred_slice)
        except Exception:
            r2 = np.nan
            
        try:
            kge_val = kge(y_true_slice, y_pred_slice)
        except Exception:
            kge_val = np.nan
            
        results[step] = {
            'rmse': rmse,
            'mae': mae,
            'nse': nse_val,
            'r2': r2,
            'kge': kge_val
        }
        
    return results



########################################################################################################################
#                                                                  
# TEST SECTION
#
########################################################################################################################
if __name__ == "__main__":
    # Test KGE with example data;
    logger.info("Testing KGE metric with example data:")
    
    # Example 1: Perfect prediction (should be close to 1.0);
    y_true_1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    kge_1 = kge(y_true=y_true_1, y_pred=y_pred_1)
    logger.info("Example 1 - Perfect prediction:")
    logger.info("  y_true: %s", y_true_1)
    logger.info("  y_pred: %s", y_pred_1)
    logger.info("  KGE: %s", kge_1)
    
    # Example 2: Predictions with some error;
    y_true_2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_2 = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
    kge_2 = kge(y_true=y_true_2, y_pred=y_pred_2)
    logger.info("Example 2 - Predictions with error:")
    logger.info("  y_true: %s", y_true_2)
    logger.info("  y_pred: %s", y_pred_2)
    logger.info("  KGE: %s", kge_2)
    
    # Example 3: Larger dataset with noise;
    np.random.seed(42)
    y_true_3 = np.random.randn(20) * 10 + 50  # Mean=50, std=10;
    y_pred_3 = y_true_3 + np.random.randn(20) * 2  # Add some noise;
    kge_3 = kge(y_true=y_true_3, y_pred=y_pred_3)
    logger.info("Example 3 - Larger dataset with noise:")
    logger.info("  y_true: %s ... (showing first 5 of %s)", y_true_3[:5], len(y_true_3))
    logger.info("  y_pred: %s ... (showing first 5 of %s)", y_pred_3[:5], len(y_pred_3))
    logger.info("  KGE: %s", kge_3)
    
    # Example 4: Different scales;
    y_true_4 = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred_4 = np.array([100.0, 200.0, 300.0, 750.0, 750.0])
    kge_4 = kge(y_true=y_true_4, y_pred=y_pred_4)
    logger.info("Example 4 - Different scales:")
    logger.info("  y_true: %s", y_true_4)
    logger.info("  y_pred: %s", y_pred_4)
    logger.info("  KGE: %s", kge_4)