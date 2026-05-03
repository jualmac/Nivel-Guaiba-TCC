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