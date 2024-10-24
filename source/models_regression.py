"""
Implementation of the different Regression Models, ARIMA Models and Decision Tree Models in the form of classes. These 
models are meant to be used side-by-side with one another. The intention is to be able to compare the results of these 
models and quickly select the best one for each situation. For these reasons, the output of all models are in the same 
format, allowing for easy concatenation of the dataframes and comparision;
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import math
import numpy as np
import pandas as pd
from ray import tune
from typing import Tuple, List, Optional, Any, Dict, Type
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor

########################################################################################################################
#
# MODELS
#
########################################################################################################################
class BaseRegressionModel:
    def __init__(self) -> None:
        """
        Initializes the base regression model;

        Attributes:
            model: The regression model instance;
            best_model: The best model after training or hyperparameter tuning;
            grid_search: The grid search object for hyperparameter tuning;
        """
        self.model = None
        self.best_model = None
        self.grid_search = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the regression model on the provided training data;

        Parameters:
            - X_train: The feature set for training;
            - y_train: The target variable for training;

        Raises:
            NotImplementedError: This method should be implemented in subclasses;
        """
        raise NotImplementedError

    def predict(self, X_test: pd.DataFrame, X_test_index: Optional[pd.Index] = None) -> pd.DataFrame:
        """
        Makes predictions using the trained model;

        Parameters:
            - X_test: The feature set for which predictions are to be made;
            - X_test_index (optional): Index to use for the resulting DataFrame. If None, the index of X_test is used;

        Returns:
            - pd.DataFrame: A DataFrame containing the predictions;
        """
        if self.best_model is None:
            raise RuntimeError("Model has not been trained yet.")
        y_pred = self.best_model.predict(X_test)

        index_to_use = X_test_index if X_test_index is not None else X_test.index
        return pd.DataFrame(y_pred, index=index_to_use, columns=[self.__class__.__name__])

    def set_params(self, **params: Any) -> None:
        """Sets the parameters of the model"""
        self.model.set_params(**params)

class LinearModel(BaseRegressionModel):
    def __init__(self) -> None:
        """Initializes the Linear Regression model"""
        super().__init__()
        self.model = LinearRegression()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the Linear Regression model on the provided training data"""
        self.model.fit(X_train, y_train)
        self.best_model = self.model

class LassoModel(BaseRegressionModel):
    def __init__(self, param_grid: Optional[dict] = None, n_splits: int = 10) -> None:
        """
        Initializes the Lasso Regression model with hyperparameter tuning;

        Parameters:
            - param_grid (optional): Dictionary of parameters for grid search;
            - n_splits (int): Number of splits for time series cross-validation;
        """
        super().__init__()
        self.model = Lasso(max_iter=10000)
        self.param_grid = param_grid or {'alpha': np.arange(0.1, 10.1, 0.1)}
        self.n_splits = n_splits

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the Lasso Regression model with hyperparameter tuning using GridSearchCV;"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=tscv,
                                         scoring='neg_root_mean_squared_error')
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_

class RidgeModel(LassoModel):
    def __init__(self, param_grid: Optional[dict] = None, n_splits: int = 10) -> None:
        """
        Initializes the Ridge Regression model with hyperparameter tuning;

        Parameters:
            - param_grid (optional): Dictionary of parameters for grid search;
            - n_splits (int): Number of splits for time series cross-validation;
        """
        super().__init__(param_grid, n_splits)
        self.model = Ridge()

class ElasticNetModel(LassoModel):
    def __init__(self, param_grid: Optional[dict] = None, n_splits: int = 10) -> None:
        """
        Initializes the ElasticNet Regression model with hyperparameter tuning;

        Parameters:
            - param_grid (optional): Dictionary of parameters for grid search;
            - n_splits (int): Number of splits for time series cross-validation;
        """
        super().__init__(param_grid, n_splits)
        self.model = ElasticNet()
        self.param_grid = param_grid or {
            'alpha': np.arange(0.1, 10.1, 0.1),
            'l1_ratio': np.arange(0.1, 1.1, 0.1)
        }

class PolynomialLinearModel(BaseRegressionModel):
    def __init__(self, degrees: Optional[List[int]] = None, n_splits: int = 10) -> None:
        """
        Initializes the Polynomial Linear Regression model;

        Parameters:
            - degrees (optional): List of degrees to consider for polynomial features;
            - n_splits (int): Number of splits for time series cross-validation;
        """
        super().__init__()
        self.degrees = degrees if degrees else [2, 3, 4]
        self.n_splits = n_splits
        self.poly_features = None  # Initialize as None
        self.model = LinearRegression()

    def find_best_degree(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Finds the best polynomial degree based on cross-validated RMSE;

        Parameters:
            - X: The feature set;
            - y: The target variable;

        Returns (Updates):
            - best_degree: The best polynomial degree found;
            - poly_features: The PolynomialFeatures instance for the best degree;
        """
        best_score = float('inf')
        for degree in self.degrees:
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            score = 0
            for train_index, val_index in tscv.split(X):
                X_train, x_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                # Create and fit PolynomialFeatures for the current degree;
                poly_features = PolynomialFeatures(degree=degree)
                X_train_poly = poly_features.fit_transform(X_train)
                
                # Fit the model on the polynomial features;
                self.model.fit(X_train_poly, y_train)
                
                # Transform the validation set using the fitted PolynomialFeatures;
                prediction_val = self.model.predict(poly_features.transform(x_val))
                score += root_mean_squared_error(y_true=y_val, y_pred=prediction_val)

            score /= tscv.get_n_splits()

            if score < best_score:
                best_score = score
                self.best_degree = degree
                self.poly_features = PolynomialFeatures(degree=degree)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the Polynomial Linear Regression model by finding the best degree and fitting the model on the polynomial 
        features;

        Parameters:
            - X_train: The feature set for training;
            - y_train: The target variable for training;
        """
        self.find_best_degree(X_train, y_train)
        
        # Fit PolynomialFeatures using the best degree;
        self.poly_features = PolynomialFeatures(degree=self.best_degree)
        X_poly = self.poly_features.fit_transform(X_train)
        self.model.fit(X_poly, y_train)
        self.best_model = self.model

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained polynomial regression model;

        Parameters:
            - X_test: The feature set for which predictions are to be made;

        Returns:
            - pd.DataFrame: A DataFrame containing the predictions;

        Raises:
            RuntimeError: If the best degree has not been found;
        """
        if self.best_degree is None:
            raise RuntimeError("Best degree has not been found.")
        
        # Transform the test data using the fitted PolynomialFeatures
        X_test_poly = self.poly_features.transform(X_test)
        print(self.best_degree)
        return super().predict(X_test=X_test_poly, X_test_index=X_test.index)

class ARIMAModel:
    def __init__(self) -> None:
        self.model = None
        self.model_fit = None
        self.best_order = None

    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                 p_values: range = range(2), 
                 d_values: range = range(2), 
                 q_values: range = range(2)) -> Tuple[int, int, int]:
        """
        Evaluates combinations of p, d, and q values to find the best ARIMA model;

        Parameters:
            - X: The feature set;
            - y: The target variable
            - p_values: Range of values for the autoregressive term;
            - d_values: Range of values for the differencing term;
            - q_values: Range of values for the moving average term;

        Returns:
            - best_cfg: Tuple with the best (p, d, q) configuration;
        """
        # Train and Test division
        division_size = int(0.1 * len(X))
        X_train, y_train = X.iloc[:-division_size], y.iloc[:-division_size]
        X_test, y_test = X.iloc[-division_size:], y.iloc[-division_size:]

        best_score = float("inf")
        best_cfg = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    rmse = self.evaluate_order(X_train, y_train, X_test, y_test, order)
                    
                    if rmse is not None and rmse < best_score:
                        best_score, best_cfg = rmse, order
        
        print(f'Best ARIMA Order: {best_cfg} with RMSE: {best_score:.4f}')
        self.best_order = best_cfg
        return best_cfg

    def evaluate_order(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series, 
                       order: Tuple[int, int, int]) -> Optional[float]:
        """
        Fits the ARIMA model for a specific order and returns the RMSE;

        Parameters:
            - X_train: Training feature set;
            - y_train: Training target variable;
            - X_test: Testing feature set;
            - y_test: Testing target variable;
            - order: Tuple of (p, d, q) for the ARIMA model;

        Returns:
            - RMSE of the predictions made by the ARIMA model;
        """
        try:
            self.train(X_train, y_train, custom_order=order)
            y_pred = self.predict(X_test)
            return root_mean_squared_error(y_test, y_pred['ARIMA'])
        except Exception as e:
            print(f"Error with order {order}: {e}")
            return None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              custom_order: Tuple[int, int, int] = (5, 1, 1), 
              order_range: Optional[Tuple[int, int, int]] = None) -> None:
        """Fits the ARIMA model to the training data. If order_range is provided, it evaluates possible orders to find the best one"""
        if order_range is not None:
            best_order = self.evaluate(X=X_train, y=y_train, p_values=range(order_range[0]), d_values=range(order_range[1]), q_values=range(order_range[2]))
        else:
            best_order = custom_order

        self.model = ARIMA(endog=y_train, exog=X_train, order=best_order)
        self.model_fit = self.model.fit()

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Predicts values using the fitted ARIMA model"""
        if self.model_fit is None:
            raise RuntimeError("Model has not been fitted yet. Call `fit` before `predict`.")

        y_pred = self.model_fit.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)
        y_pred = y_pred.rename_axis('date').rename('ARIMA').reset_index()
        y_pred.set_index('date', inplace=True)
        return y_pred

class DecisionTreeModel:
    #TODO FILL WITH DEFAULT VALUES;
    default_param_grid = {
        'max_features': [None],                                                     #How many features are being used. 'None' will use n_features provided;
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30],                            #Minimum number of samples that a node must have before being splited;
        'min_samples_leaf': [5, 6, 7, 8, 9, 10],                                    #Minimum number of samples required to be at a leaf node;
        'max_depth': [4, 5],                                                        #Determines the max distance between a branch of the tree and the leaves;
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse']             #Function that mesures the quality of the split;
    }

    def __init__(self, param_grid: Optional[Dict[str, Any]] = None) -> None:
        self.model = DecisionTreeRegressor()
        self.grid_search = None
        self.best_model = None
        
        # Use the default parameter grid if none is provided
        self.param_grid = param_grid if param_grid is not None else self.default_param_grid

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Decision Tree model using GridSearchCV;
        
        Parameters:
            - X_train (pandas DataFrame): Training features;
            - y_train: (pandas DataFrame): Training target;
        """
        tscv = TimeSeriesSplit(n_splits=10)
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=tscv,
            scoring='neg_root_mean_squared_error'
        )
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the trained model
        
        Parameters:
            - X_test (pandas DataFrame): Test features;
        
        Returns:
            - y_pred (pandas DataFrame): Predictions with timestamp index;
        """
        if self.best_model is None:
            raise RuntimeError("Model has not been trained yet. Call `train` before `predict`.")
        
        y_pred = self.best_model.predict(X_test)
        y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=['DecisionTree'])
        y_pred.index.name = 'date'
        return y_pred

def run_regression_model(model_class: Type[BaseRegressionModel], X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Instantiates a regression model, trains it on the provided training data, 
    and makes predictions on the test data.

    Parameters:
        - model_class (Type[BaseRegressionModel]): The class of the regression model to be instantiated.
        - X_train (pandas DataFrame): The training feature set.
        - y_train (pandas Series): The training target variable.
        - X_test (pandas DataFrame): The test feature set.

    Returns:
        - predictions (pandas DataFrame): A DataFrame containing the predictions made by the model on the test set.
    """
    model = model_class()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def run_all_regression_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Runs multiple regression models in parallel and aggregates their predictions;

    Parameters:
        - X_train (pandas DataFrame): The training feature set;
        - y_train (pandas Series): The training target variable;
        - X_test (pandas DataFrame): The test feature set;

    Returns:
        - final_results (pandas DataFrame): A DataFrame containing the predictions from all models, concatenated along the columns;
    """
    models = [
        LinearModel,
        LassoModel,
        RidgeModel,
        ElasticNetModel,
        PolynomialLinearModel,
        ARIMAModel,
        DecisionTreeModel
    ]

    results = []

    # Run simple models in paralel;
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_regression_model, model, X_train, y_train, X_test): model.__name__ for model in models}

        for future in futures:
            model_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error running model {model_name}: {e}")
    final_results = pd.concat(results, axis=1)
    return final_results