import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import os
from datetime import datetime

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

    def log_model(self, model, artifact_path: str = "model"):
        """
        Log a scikit-learn compatible model.
        
        Parameters:
            model: The model object (must be compatible with mlflow.sklearn).
            artifact_path: Path within the run to store the model.
        """
        mlflow.sklearn.log_model(model, artifact_path)

    def end_run(self):
        """End the current run."""
        mlflow.end_run()