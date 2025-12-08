"""
This file includes alternative metrics to be used in statistical analysis. These metrics are not allways includes in 
packages like scikit learn. However, some of them may depend on these packages on someway;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
from permetrics.regression import RegressionMetric

########################################################################################################################
#                                                                  
# FUNCTIONS
#
########################################################################################################################
def relative_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Relative Root Mean Squared Error (RRMSE), an error metric related to the RMSE. It computes the 
    Root Mean Squared Error and normalizes it by the range of the predicted values. The result is expressed as a 
    percentage, making it suitable for comparison across variables on different scales;

    Parameters:
        y_true (np.ndarray): Actual values of the dependent variable;
        y_pred (np.ndarray): Predicted values of the dependent variable;

    Returns:
        float: The RRMSE value expressed as a percentage for comparison;
    """
    n = len(y_true)
    num = np.sum(np.square(y_true - y_pred))/n
    den = np.sum(np.square(y_pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss


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


########################################################################################################################
#                                                                  
# TEST SECTION
#
########################################################################################################################
if __name__ == "__main__":
    # Test KGE with example data;
    print("[metrics.py] Testing KGE metric with example data:\n")
    
    # Example 1: Perfect prediction (should be close to 1.0);
    y_true_1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    kge_1 = kge(y_true=y_true_1, y_pred=y_pred_1)
    print(f"[metrics.py] Example 1 - Perfect prediction:")
    print(f"[metrics.py]   y_true: {y_true_1}")
    print(f"[metrics.py]   y_pred: {y_pred_1}")
    print(f"[metrics.py]   KGE: {kge_1}\n")
    
    # Example 2: Predictions with some error;
    y_true_2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_2 = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
    kge_2 = kge(y_true=y_true_2, y_pred=y_pred_2)
    print(f"[metrics.py] Example 2 - Predictions with error:")
    print(f"[metrics.py]   y_true: {y_true_2}")
    print(f"[metrics.py]   y_pred: {y_pred_2}")
    print(f"[metrics.py]   KGE: {kge_2}\n")
    
    # Example 3: Larger dataset with noise;
    np.random.seed(42)
    y_true_3 = np.random.randn(20) * 10 + 50  # Mean=50, std=10;
    y_pred_3 = y_true_3 + np.random.randn(20) * 2  # Add some noise;
    kge_3 = kge(y_true=y_true_3, y_pred=y_pred_3)
    print(f"[metrics.py] Example 3 - Larger dataset with noise:")
    print(f"[metrics.py]   y_true: {y_true_3[:5]} ... (showing first 5 of {len(y_true_3)})")
    print(f"[metrics.py]   y_pred: {y_pred_3[:5]} ... (showing first 5 of {len(y_pred_3)})")
    print(f"[metrics.py]   KGE: {kge_3}\n")
    
    # Example 4: Different scales;
    y_true_4 = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred_4 = np.array([100.0, 200.0, 300.0, 750.0, 750.0])
    kge_4 = kge(y_true=y_true_4, y_pred=y_pred_4)
    print(f"[metrics.py] Example 4 - Different scales:")
    print(f"[metrics.py]   y_true: {y_true_4}")
    print(f"[metrics.py]   y_pred: {y_pred_4}")
    print(f"[metrics.py]   KGE: {kge_4}")