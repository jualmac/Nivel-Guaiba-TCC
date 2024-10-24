"""
Data transfomations that can be used in data modeling;
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pandas.core.frame import DataFrame
import numpy as np

########################################################################################################################
#
# TRANSFORMATIONS
#
########################################################################################################################
def nature_encode(df: DataFrame, col: str, div_period: int):
    """
    Applies a Nature Cyclical Transformation, where each period is a combination of sin and cos;

    Parameters:
        - df (pandas DataFrame): DataFrame on which the function will be applied;
        - col (str): Period column on which the function will be applied;
        - div_period (int): Amount of periods until the cycle restarts (e.g. month=12, week=7, etc);
    """
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / div_period)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / div_period)
    return None