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