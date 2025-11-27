"""
Utility functions for logging experiments with MLflow;
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import os
from datetime import datetime

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
class MLFlowHandler:
    def __init__(self, experiment_name: str = "river_level_forecasting", tracking_uri: Optional[str] = None):
        """
        Initialize MLFlow handler.
        
        Parameters:
            experiment_name (str): Name of the experiment in MLFlow.
            tracking_uri (str): URI for MLFlow tracking server (default: local ./mlruns).
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLFlow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log a dictionary of parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log a dictionary of metrics."""
        mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path: str = "model", input_example: Optional[Any] = None):
        """
        Log a scikit-learn compatible model.
        
        Parameters:
            model: The model object (must be compatible with mlflow.sklearn).
            artifact_path: Path within the run to store the model.
            input_example: Optional input example for model signature inference.
        """
        mlflow.sklearn.log_model(model, artifact_path, input_example=input_example)

    def end_run(self):
        """End the current run."""
        mlflow.end_run()

    def load_best_params(self, metric_name: str = "score", mode: str = "max") -> Dict[str, Any]:
        """
        Retrieve parameters from the best run based on a metric.
        
        Parameters:
            metric_name (str): Name of the metric to sort by.
            mode (str): 'max' (higher is better) or 'min' (lower is better).
            
        Returns:
            Dict[str, Any]: Dictionary of parameters from the best run.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {}
            
            order_by = [f"metrics.{metric_name} DESC"] if mode == "max" else [f"metrics.{metric_name} ASC"]
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=order_by,
                max_results=1
            )
            
            if runs.empty:
                return {}
                
            # Return params from the first (best) run -> params are stored with prefix 'params.' in the dataframe
            best_run = runs.iloc[0]

            # Filter out NaN values and clean keys
            params = {
                k.replace("params.", ""): v 
                for k, v in best_run.items() 
                if k.startswith("params.") and v is not None and str(v) != "nan"
            }
            return params
        except Exception as e:
            print(f"Error retrieving best params: {e}")
            return {}