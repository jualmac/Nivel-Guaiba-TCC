"""
Main backend orchestrator for model training and evaluation;

Two-mode architecture:
  - General (Reliability): Nested Cross-Validation (Walk-Forward) on the FULL dataset;
  - Case Study (Capability): Isolated May 2024 flood event evaluation using best hyperparameters;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import sys

# Add project root to sys.path to allow absolute imports from 'src' when run as a script;
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Optional, List
from src.db_handler import DBConnection
from src.pipelines.pipeline_preparation import encoding_pipeline
from src.models.train_models import training_pipeline
from src.mlflow_utils import MLFlowHandler
from src.etl.data_io import save_to_database
from src.pipelines.pipeline_data import pipeline_data
from src.etl.data_imputation import feature_imputation
from src.util import configure_logging
from src.metrics import evaluate_multi_horizon

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# HELPER: TRAIN AND EVALUATE A SINGLE MODEL
#
########################################################################################################################
def _train_and_predict(name, pipeline, X_train, y_train, X_val, y_val, X_test,
                       optimize, early_stopping, inner_cv_folds=5):
    """Train a single model pipeline and return predictions on X_test."""
    feature_steps = pipeline.steps[:-1]
    feature_pipeline = Pipeline(feature_steps)

    if name in ['XGBOOST', 'LIGHTGBM', 'LSTM']:
        model_step_name, model_instance = pipeline.steps[-1]
        feature_pipeline.fit(X_train, y_train)

        X_train_processed = feature_pipeline.transform(X_train)
        X_val_processed = feature_pipeline.transform(X_val)

        model_params = {
            "optimize_hyperparameters": optimize,
            "X_val": X_val_processed,
            "y_val": y_val,
            "early_stopping": early_stopping,
            "feature_pipeline": feature_pipeline,
            "X_raw": X_train,
            "cv_n_splits": inner_cv_folds,
            "cv_gap": 24,
        }

        model_instance.fit(X_train_processed, y_train, **model_params)
        pipeline = Pipeline(feature_pipeline.steps + [(model_step_name, model_instance)])
    else:
        fit_params = {f"{name}__optimize_hyperparameters": optimize}
        pipeline.fit(X_train, y_train, **fit_params)

    y_pred = pipeline.predict(X_test)
    return y_pred, pipeline


########################################################################################################################
#                                                                  
# GENERAL MODE: NESTED CROSS-VALIDATION (Walk-Forward) ON FULL DATASET
#
########################################################################################################################
def _run_general_mode(
    df, target_column, models_to_use, steps, cv_folds,
    random_state, n_trials, batch, freq, mode, optimize,
    early_stopping, use_lags, use_rolling_stats, use_cumulative,
    use_feature_selection, n_features, mlflow_handler, log_mlflow
):
    """
    Reliability evaluation: Nested CV (Walk-Forward) using the FULL dataset;
    Returns aggregated metrics DataFrame and per-fold metrics DataFrame;
    """
    logger.info("==================== GENERAL MODE: NESTED CROSS-VALIDATION ====================")
    
    max_horizon = max(steps)
    N = len(df)

    if N < max_horizon * (cv_folds + 2):
        logger.warning("Dataset may be too small for the requested folds and horizons.")

    all_metrics = []
    all_test_predictions = []

    for fold in range(cv_folds):
        logger.info("================================== FOLD %s / %s ==================================", fold+1, cv_folds)

        # Chronological split for the fold;
        test_end = N - (cv_folds - 1 - fold) * max_horizon
        test_start = test_end - max_horizon

        # Validation for early stopping: last max_horizon of the training fold;
        val_end = test_start
        val_start = val_end - max_horizon
        train_end = val_start

        df_train_fold = df.iloc[:train_end].copy()
        df_val_fold = df.iloc[val_start:val_end].copy()
        df_test_fold = df.iloc[test_start:test_end].copy()

        logger.info("Fold %s lengths - Train: %s, Val: %s, Test: %s", fold+1, len(df_train_fold), len(df_val_fold), len(df_test_fold))

        X_train = df_train_fold.drop(columns=[target_column])
        y_train = df_train_fold[target_column]
        X_val = df_val_fold.drop(columns=[target_column])
        y_val = df_val_fold[target_column]
        X_test = df_test_fold.drop(columns=[target_column])
        y_test = df_test_fold[target_column]

        # Imputation;
        X_train = feature_imputation(X_train)
        X_val = feature_imputation(X_val)
        X_test = feature_imputation(X_test)

        # Preprocessor;
        preprocessor = encoding_pipeline(target_column=target_column)
        preprocessor.fit(X_train, y_train)

        # Training pipelines;
        pipelines = training_pipeline(
            preprocessor=preprocessor,
            models_to_use=models_to_use,
            random_state=random_state,
            n_trials=n_trials,
            batch=batch,
            freq=freq,
            mode=mode,
            use_lags=use_lags,
            use_rolling_stats=use_rolling_stats,
            use_cumulative=use_cumulative,
            use_feature_selection=use_feature_selection,
            n_features=n_features
        )

        for name, pipeline in pipelines.items():
            logger.info("Training %s for Fold %s...", name, fold+1)

            if log_mlflow:
                mlflow_handler.start_run(run_name=f"general_{name}_fold_{fold+1}")
                mlflow_handler.log_params({"model_type": name, "fold": fold + 1, "pipeline_mode": "general"})

            y_pred, _ = _train_and_predict(
                name, pipeline, X_train, y_train, X_val, y_val, X_test,
                optimize=optimize, early_stopping=early_stopping
            )

            # Inverse log-transform predictions and true values back to original scale;
            y_pred_orig = np.expm1(y_pred)
            y_test_orig = np.expm1(y_test.values)

            # Evaluate multi-horizon (original scale);
            horizon_metrics = evaluate_multi_horizon(y_test_orig, y_pred_orig, steps)

            for step, h_metrics in horizon_metrics.items():
                all_metrics.append({
                    'model_name': name,
                    'target_column': target_column,
                    'fold': fold + 1,
                    'step': step,
                    'rmse': h_metrics['rmse'],
                    'mae': h_metrics['mae'],
                    'nse': h_metrics['nse'],
                    'r2': h_metrics['r2'],
                    'kge': h_metrics['kge']
                })

            # Collect test predictions for DB;
            date_col = None
            if "Data_Hora_Medicao" in X_test.columns:
                date_col = X_test["Data_Hora_Medicao"].values
            elif hasattr(X_test, "index"):
                date_col = X_test.index.values

            for step in steps:
                # Each step covers the first `step` hours of the test fold;
                n = min(step, len(y_test_orig))
                pred_df = pd.DataFrame({
                    'model_name': name,
                    'fold': fold + 1,
                    'step': step,
                    'y_true': y_test_orig[:n],
                    'y_pred': y_pred_orig[:n],
                    'y_pred_log': y_pred[:n],
                    'target_column': target_column
                })
                if date_col is not None:
                    pred_df['date'] = date_col[:n]
                all_test_predictions.append(pred_df)

            if log_mlflow:
                mlflow_handler.end_run()

    # Aggregate results;
    df_metrics = pd.DataFrame(all_metrics)
    agg_metrics_list = []

    if not df_metrics.empty:
        agg_cols = ['rmse', 'mae', 'nse', 'r2', 'kge']
        grouped = df_metrics.groupby(['model_name', 'target_column', 'step'])

        for (model, target, step), group in grouped:
            agg_row = {'model_name': model, 'target_column': target, 'step': step}
            for col in agg_cols:
                agg_row[col] = group[col].mean()
                agg_row[f"{col}_std"] = group[col].std()
            agg_metrics_list.append(agg_row)

        df_agg_metrics = pd.DataFrame(agg_metrics_list)
        logger.info("\n=== Nested CV Aggregated Results ===\n%s", df_agg_metrics)
    else:
        df_agg_metrics = pd.DataFrame()

    df_test_predictions = pd.concat(all_test_predictions, ignore_index=True) if all_test_predictions else None
    return df_agg_metrics, df_test_predictions


########################################################################################################################
#                                                                  
# CASE STUDY MODE: MAY 2024 FLOOD EVENT (Capability)
#
########################################################################################################################
def _run_case_study_mode(
    df, target_column, models_to_use, steps,
    random_state, n_trials, batch, freq, mode,
    early_stopping, save_to_db, use_lags, use_rolling_stats, use_cumulative,
    use_feature_selection, n_features, mlflow_handler, log_mlflow
):
    """
    Capability evaluation: Train on all data before May 2024, test on May 2024 flood;
    Uses optimize=False to load best hyperparameters from Nested CV runs;
    Returns case study metrics, predictions, and features DataFrames;
    """
    logger.info("==================== CASE STUDY MODE: SPECIFIC FLOOD ====================")

    case_study_start = pd.to_datetime('2025-06-20')
    case_study_end = pd.to_datetime('2025-07-01')

    df_train = df[df['Data_Hora_Medicao'] < case_study_start].copy().reset_index(drop=True)
    df_test = df[(df['Data_Hora_Medicao'] >= case_study_start) & (df['Data_Hora_Medicao'] < case_study_end)].copy().reset_index(drop=True)

    logger.info("Case Study - Training data: %s rows, Test data: %s rows", len(df_train), len(df_test))

    if len(df_test) == 0:
        logger.warning("No test data available for May 2024 case study. Skipping.")
        return pd.DataFrame(), None, None

    max_horizon = max(steps)

    X_train_cs = df_train.drop(columns=[target_column])
    y_train_cs = df_train[target_column]

    # Use last max_horizon from training as validation set for early stopping;
    val_start = len(X_train_cs) - max_horizon
    X_val_cs = X_train_cs.iloc[val_start:].copy()
    y_val_cs = y_train_cs.iloc[val_start:].copy()
    X_train_cs = X_train_cs.iloc[:val_start].copy()
    y_train_cs = y_train_cs.iloc[:val_start].copy()

    X_test_cs = df_test.drop(columns=[target_column])
    y_test_cs = df_test[target_column]

    # Imputation;
    X_train_cs = feature_imputation(X_train_cs)
    X_val_cs = feature_imputation(X_val_cs)
    X_test_cs = feature_imputation(X_test_cs)

    # Preprocessor;
    preprocessor_cs = encoding_pipeline(target_column=target_column)
    preprocessor_cs.fit(X_train_cs, y_train_cs)

    pipelines_cs = training_pipeline(
        preprocessor=preprocessor_cs,
        models_to_use=models_to_use,
        random_state=random_state,
        n_trials=n_trials,
        batch=batch,
        freq=freq,
        mode=mode,
        use_lags=use_lags,
        use_rolling_stats=use_rolling_stats,
        use_cumulative=use_cumulative,
        use_feature_selection=use_feature_selection,
        n_features=n_features
    )

    all_predictions_cs = []
    case_study_metrics = []
    all_features_cs = []

    for name, pipeline in pipelines_cs.items():
        logger.info("Training %s for Case Study...", name)

        if log_mlflow:
            mlflow_handler.start_run(run_name=f"case_study_{name}")
            mlflow_handler.log_params({"model_type": name, "pipeline_mode": "case_study"})

        # optimize=False -> load best hyperparameters from Nested CV;
        y_pred, fitted_pipeline = _train_and_predict(
            name, pipeline, X_train_cs, y_train_cs, X_val_cs, y_val_cs, X_test_cs,
            optimize=False, early_stopping=early_stopping
        )

        # Inverse log-transform predictions and true values back to original scale;
        y_pred_log = y_pred.copy()          # keep log-scale predictions for DB
        y_pred = np.expm1(y_pred)
        y_test_cs_orig = np.expm1(y_test_cs.values)

        # Collect features for DB;
        if save_to_db:
            def collect_features(split_name, X_split, y_split):
                orig_index = X_split.index if hasattr(X_split, "index") else None
                features = X_split
                for step_name, step_transformer in fitted_pipeline.steps[:-1]:
                    features = step_transformer.transform(features)
                if not isinstance(features, pd.DataFrame):
                    features = pd.DataFrame(features, index=orig_index)
                features = features.copy()
                target_series = y_split.reindex(features.index) if isinstance(y_split, pd.Series) else pd.Series(y_split, index=features.index, name='target')
                features['target'] = target_series
                features['model_name'] = name
                features['split'] = split_name
                return features

            all_features_cs.append(collect_features('train', X_train_cs, y_train_cs))
            all_features_cs.append(collect_features('val', X_val_cs, y_val_cs))
            all_features_cs.append(collect_features('test', X_test_cs, y_test_cs))

        # Build predictions DataFrame;
        date_col = None
        if "Data_Hora_Medicao" in X_test_cs.columns:
            date_col = X_test_cs["Data_Hora_Medicao"].copy()
        elif hasattr(X_test_cs, "index"):
            date_col = X_test_cs.index

        pred_df = pd.DataFrame({
            'model_name': name,
            'y_true': y_test_cs_orig,       # original scale
            'y_pred': y_pred,               # original scale (after expm1)
            'y_pred_log': y_pred_log,       # log scale (before expm1)
            'target_column': target_column
        })
        if hasattr(X_test_cs, 'index'):
            pred_df.index = X_test_cs.index
        if date_col is not None:
            pred_df['date'] = date_col

        all_predictions_cs.append(pred_df)

        # Evaluate multi-horizon (original scale);
        horizon_metrics = evaluate_multi_horizon(y_test_cs_orig, y_pred, steps)
        for step, h_metrics in horizon_metrics.items():
            case_study_metrics.append({
                'model_name': f"{name}_CASE_STUDY",
                'target_column': target_column,
                'step': step,
                'rmse': h_metrics['rmse'],
                'mae': h_metrics['mae'],
                'nse': h_metrics['nse'],
                'r2': h_metrics['r2'],
                'kge': h_metrics['kge']
            })

        if log_mlflow:
            mlflow_handler.end_run()

    df_cs_metrics = pd.DataFrame(case_study_metrics)
    df_predictions = None
    if all_predictions_cs:
        full_preds = pd.concat(all_predictions_cs, ignore_index=False)
        if 'date' in full_preds.columns:
            df_predictions = full_preds.pivot(index='date', columns='model_name', values='y_pred')
            y_true = full_preds.groupby('date')['y_true'].first()
            df_predictions['y_true'] = y_true
            df_predictions = df_predictions.reset_index()
        else:
            df_predictions = full_preds.pivot_table(index=full_preds.index, columns='model_name', values='y_pred')
            y_true = full_preds.groupby(level=0)['y_true'].first()
            df_predictions['y_true'] = y_true
            df_predictions = df_predictions.reset_index().rename(columns={'index': 'date'})

    df_features = pd.concat(all_features_cs, ignore_index=True) if all_features_cs else None
    return df_cs_metrics, df_predictions, df_features


########################################################################################################################
#                                                                  
# MAIN BACKEND PIPELINE
#
########################################################################################################################
def main(
    target_column: str,
    pipeline_mode: str = 'all',
    models_to_use: Optional[list] = None,
    steps: List[int] = [24, 72, 168, 360],
    cv_folds: int = 5,
    random_state: int = 42,
    n_trials: int = 10,
    batch: int = 128,
    freq: str = 'h',
    mode: str = 'CPU',
    optimize: bool = True,
    early_stopping: int = 50,
    save_to_db: bool = False,
    use_lags: bool = False,
    use_rolling_stats: bool = False,
    use_cumulative: bool = False,
    use_feature_selection: bool = False,
    n_features: Optional[int] = None,
    log_mlflow: bool = True
) -> None:
    """
    Main orchestrator for model training and evaluation;
    
    Parameters:
        pipeline_mode (str): Execution mode - 'general' (Nested CV on full data),
            'case' (May 2024 flood study), or 'all' (both sequentially);
    """
    # Initialize MLFlow Handler;
    mlflow_handler = None
    if log_mlflow:
        mlflow_handler = MLFlowHandler(experiment_name="river_level_forecasting")
        mlflow_handler.start_run(run_name="initial_config")

        log_params = {
            "target_column": target_column,
            "pipeline_mode": pipeline_mode,
            "models_to_use": str(models_to_use) if models_to_use is not None else "all",
            "steps": str(steps),
            "cv_folds": cv_folds,
            "random_state": random_state,
            "n_trials": n_trials,
            "batch": batch,
            "freq": freq,
            "mode": mode,
            "optimize": optimize,
            "early_stopping": early_stopping,
            "save_to_db": save_to_db,
            "use_lags": use_lags,
            "use_rolling_stats": use_rolling_stats,
            "use_cumulative": use_cumulative,
            "use_feature_selection": use_feature_selection,
            "n_features": n_features if n_features is not None else "None"
        }
        mlflow_handler.log_params(log_params)
        mlflow_handler.end_run()

    # Execute data pipeline and read clean data;
    logger.info("Executing ETL pipeline...")
    df = pipeline_data(save_to_db=save_to_db)
    logger.info("Loaded %s rows from data pipeline", len(df))

    # Enforce chronological order;
    if "Data_Hora_Medicao" in df.columns:
        df = df.sort_values("Data_Hora_Medicao").reset_index(drop=True)
    else:
        logger.warning("Column 'Data_Hora_Medicao' not found; skipping chronological sort;")

    # Drop missing targets;
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    df['Data_Hora_Medicao'] = pd.to_datetime(df['Data_Hora_Medicao'])

    # Log-transform the target: y_log = ln(y + 1)
    # Models train on log-scaled target; predictions are inverse-transformed after predict();
    df[target_column] = np.log1p(df[target_column])
    logger.info("Applied log1p transform to target '%s'", target_column)

    logger.info("Full dataset: %s rows (%s to %s)", len(df),
                df['Data_Hora_Medicao'].min(), df['Data_Hora_Medicao'].max())

    # Shared model training kwargs;
    model_kwargs = dict(
        models_to_use=models_to_use, steps=steps,
        random_state=random_state, n_trials=n_trials, batch=batch,
        freq=freq, mode=mode, early_stopping=early_stopping,
        use_lags=use_lags, use_rolling_stats=use_rolling_stats,
        use_cumulative=use_cumulative, use_feature_selection=use_feature_selection,
        n_features=n_features, mlflow_handler=mlflow_handler, log_mlflow=log_mlflow
    )

    df_agg_metrics = pd.DataFrame()
    df_cs_metrics = pd.DataFrame()
    df_predictions = None
    df_features = None
    df_test_predictions = None

    # ── GENERAL MODE (Reliability): Nested CV on full dataset ──
    if pipeline_mode in ('general', 'all'):
        df_agg_metrics, df_test_predictions = _run_general_mode(
            df=df, target_column=target_column,
            cv_folds=cv_folds, optimize=optimize,
            **model_kwargs
        )

    # ── CASE STUDY MODE (Capability): May 2024 flood event ──
    if pipeline_mode in ('case', 'all'):
        df_cs_metrics, df_predictions, df_features = _run_case_study_mode(
            df=df, target_column=target_column,
            save_to_db=save_to_db,
            **model_kwargs
        )

    # Save results to DB;
    if save_to_db:
        logger.info("Saving ML results to database...")
        metrics_parts = [m for m in [df_agg_metrics, df_cs_metrics] if not m.empty]
        final_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else None

        save_to_database(
            df_predictions=df_predictions,
            df_test_predictions=df_test_predictions,
            df_metrics=final_metrics,
            df_features=df_features
        )

    logger.info("Backend pipeline completed!")

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main backend training pipeline")
    parser.add_argument('--pipeline_mode', type=str, choices=['general', 'case', 'all'],
                        default='all', help="Pipeline mode: 'general' (Nested CV), 'case' (May 2024 flood), 'all' (both)")
    parser.add_argument('--steps', type=str, nargs='+', 
                        choices=['24', '72', '168', '360', '720', '1440'], 
                        default=['24', '72', '168', '360', '720'], 
                        help='Forecasting horizons in hours')
    parser.add_argument('--models_to_use', type=str, nargs='+', 
                        choices=['SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM', 'DUMMY'], 
                        default=['SARIMA', 'XGBOOST', 'LIGHTGBM', 'DUMMY'], 
                        help='List of models to train')

    parser.add_argument('--target_column', type=str, default='Cota_Adotada_87450004', help='Name of target column to predict')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of Walk-Forward folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['CPU', 'GPU', 'CUDA'], default='GPU', help='Training device mode: CPU (default), GPU (OpenCL), or CUDA')
    parser.add_argument('--batch', type=int, default=128, help='Training batch size')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--early_stopping', type=int, default=10, help='Number of rounds for early stopping (default: 50)')
    parser.add_argument('--n_features', type=int, default=30, help='Number of top features to select if use_feature_selection=True (default: 50)')
    parser.add_argument('--freq', type=str, choices=['h', 'bh', 'min', 's', 'D', 'B', 'W', 'M', 'MS', 'SMS'], default='h', help='Frequency of predictions (pandas offset)')
    
    parser.add_argument('--save_to_db', action='store_true', help='Save the results to the database')
    parser.add_argument('--no_save_to_db', dest='save_to_db', action='store_false', help='Do not save the results to the database')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter Optimization')
    parser.add_argument('--no_optimize', dest='optimize', action='store_false', help='Do not perform hyperparameter Optimization')
    parser.add_argument('--use_lags', action='store_true', help='Add lag features transformer to pipeline')
    parser.add_argument('--no_use_lags', dest='use_lags', action='store_false', help='Do not add lag features transformer')
    parser.add_argument('--use_rolling_stats', action='store_true', help='Add rolling statistics transformer to pipeline')
    parser.add_argument('--no_use_rolling_stats', dest='use_rolling_stats', action='store_false', help='Do not add rolling statistics transformer')
    parser.add_argument('--use_cumulative', action='store_true', help='Add cumulative rolling-sum transformer to pipeline')
    parser.add_argument('--no_use_cumulative', dest='use_cumulative', action='store_false', help='Do not add cumulative rolling-sum transformer')
    parser.add_argument('--use_feature_selection', action='store_true', help='Add feature selection based on RandomForest importance')
    parser.add_argument('--no_use_feature_selection', dest='use_feature_selection', action='store_false', help='Do not add feature selection')
    parser.add_argument('--log_mlflow', action='store_true', help='Log experiments to MLFlow')
    parser.add_argument('--no_log_mlflow', dest='log_mlflow', action='store_false', help='Do not log experiments to MLFlow')

    parser.set_defaults(
        save_to_db=True,
        optimize=True,
        use_lags=True,
        use_rolling_stats=True,
        use_cumulative=True,
        use_feature_selection=True,
        log_mlflow=True
    )
    
    args = parser.parse_args()
    logger.info("Arguments: %s", args)
    
    parsed_steps = [int(s) for s in args.steps]

    main(
        target_column=args.target_column,
        pipeline_mode=args.pipeline_mode,
        models_to_use=args.models_to_use,
        steps=parsed_steps,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        n_trials=args.trials,
        batch=args.batch,
        freq=args.freq,
        mode=args.mode,
        optimize=args.optimize,
        early_stopping=args.early_stopping,
        save_to_db=args.save_to_db,
        use_lags=args.use_lags,
        use_rolling_stats=args.use_rolling_stats,
        use_cumulative=args.use_cumulative,
        use_feature_selection=args.use_feature_selection,
        n_features=args.n_features,
        log_mlflow=args.log_mlflow
    )
    logger.info("All Done!")