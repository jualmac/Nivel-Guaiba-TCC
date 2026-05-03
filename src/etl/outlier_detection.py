"""
Outlier detection and removal operations;

Uses consensus-based multivariate outlier detection with ECOD and PCA algorithms;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import logging
import numpy as np
import pandas as pd
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from src.util import configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# OUTLIER DETECTION FUNCTIONS
#
########################################################################################################################
def outlier_removal(
    df: pd.DataFrame, 
    threshold_method: str = 'iqr'
) -> pd.DataFrame:
    """
    Detect and remove outliers using consensus of ECOD and PCA detectors;
    
    Applies ECOD and PCA outlier detection per station on complete rows only. Uses consensus
    approach: flags outliers only when both detectors agree. Threshold determined dynamically via
    IQR (Tukey fence) or percentile method. Replaces outliers with NaN and sets status code to 5;
    
    Parameters:
        df (pd.DataFrame): Input dataframe with station codes and feature columns;
        threshold_method (str): Threshold calculation method (default: 'iqr').
            - 'iqr': Q3 + 1.5*IQR (Tukey fence);
            - 'percentile': 95th percentile of decision scores;
    
    Returns:
        pd.DataFrame: Dataframe with outlier values set to NaN and status codes updated to 5;
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    # Prepare data;
    index_cols = ['Data_Hora_Medicao', 'codigoestacao']
    meta_cols = ['Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols + meta_cols
    feature_cols = list(set(df_cpy.columns.unique()) - set(non_feature_cols))
    
    processed_stations = []
    for station in df_cpy['codigoestacao'].unique():
        logger.info("Processing station %s...", station)
        df_station = df_cpy[df_cpy['codigoestacao'] == station].copy()
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()

        # Identify fully missing columns for this station and exclude them from outlier detection;
        fully_missing = df_station_features.isna().all()
        available_feature_cols = [col for col in feature_cols if not fully_missing[col]]
        fully_missing_cols = [col for col in feature_cols if fully_missing[col]]
        
        if fully_missing_cols:
            logger.warning("Skipping fully missing features for outlier detection: %s", fully_missing_cols)
        
        # If no features available, skip outlier detection for this station;
        if not available_feature_cols:
            logger.warning("No available features for station %s. Skipping outlier detection.", station)
            processed_stations.append(df_station)
            continue
        
        # Only use complete rows for outlier detection (considering only available features);
        df_station_available = df_station_features[available_feature_cols].copy()
        complete_mask = df_station_available.notna().all(axis=1)
        df_complete = df_station_available[complete_mask]
        
        # If no complete rows, skip outlier detection;
        if len(df_complete) == 0:
            logger.warning("No complete rows for station %s. Skipping outlier detection.", station)
            processed_stations.append(df_station)
            continue
        
        logger.info("Using %s features and %s complete rows for outlier detection", len(available_feature_cols), len(df_complete))
        
        # ECOD Detection;
        logger.info("[ECOD DETECTOR]")
        # Fit model with minimal contamination to extract decision scores (Actual threshold is determined dynamically via IQR/percentile method below);
        ecod_detector = ECOD(contamination=0.001)
        ecod_detector.fit(df_complete)
        ecod_scores = ecod_detector.decision_scores_
        
        # PCA Detection;
        logger.info("[PCA DETECTOR]")
        # Fit model with minimal contamination to extract decision scores (Actual threshold is determined dynamically via IQR/percentile method below);
        pca_detector = PCA(contamination=0.001)
        pca_detector.fit(df_complete)
        pca_scores = pca_detector.decision_scores_
        
        # Apply dynamic threshold detection based on decision scores;
        if threshold_method == 'iqr':
            # Tukey Fence (IQR method): outliers are beyond Q3 + 1.5*IQR;
            ecod_q1, ecod_q3 = np.percentile(ecod_scores, [25, 75])
            ecod_iqr = ecod_q3 - ecod_q1
            ecod_threshold = ecod_q3 + 1.5 * ecod_iqr
            ecod_predictions = (ecod_scores > ecod_threshold).astype(int)
            
            pca_q1, pca_q3 = np.percentile(pca_scores, [25, 75])
            pca_iqr = pca_q3 - pca_q1
            pca_threshold = pca_q3 + 1.5 * pca_iqr
            pca_predictions = (pca_scores > pca_threshold).astype(int)
            
            logger.info("ECOD: threshold=%.4f, outliers=%s/%s (%.2f%%)", ecod_threshold, ecod_predictions.sum(), len(ecod_predictions), 100*ecod_predictions.sum()/len(ecod_predictions))
            logger.info("PCA: threshold=%.4f, outliers=%s/%s (%.2f%%)", pca_threshold, pca_predictions.sum(), len(pca_predictions), 100*pca_predictions.sum()/len(pca_predictions))
        
        # Fixed ammount of outliers; 
        elif threshold_method == 'percentile':
            # Percentile method: use 95th percentile as threshold (top 5% are outliers);
            ecod_threshold = np.percentile(ecod_scores, 99)
            ecod_predictions = (ecod_scores > ecod_threshold).astype(int)
            
            pca_threshold = np.percentile(pca_scores, 99)
            pca_predictions = (pca_scores > pca_threshold).astype(int)
            
            logger.info("ECOD: threshold=%.4f (95th percentile), outliers=%s/%s (%.2f%%)", ecod_threshold, ecod_predictions.sum(), len(ecod_predictions), 100*ecod_predictions.sum()/len(ecod_predictions))
            logger.info("PCA: threshold=%.4f (95th percentile), outliers=%s/%s (%.2f%%)", pca_threshold, pca_predictions.sum(), len(pca_predictions), 100*pca_predictions.sum()/len(pca_predictions))
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}. Use 'iqr' or 'percentile'")

        # Combined outliers (intersection of both detectors - only flag if both agree -> Imply consensus);
        combined_outliers = (ecod_predictions & pca_predictions).astype(bool)
        logger.info("Consensus: %s/%s outliers (%.2f%%)", combined_outliers.sum(), len(combined_outliers), 100*combined_outliers.sum()/len(combined_outliers))
        
        # Map outliers back to original dataframe (only for available features);
        complete_indices = df_station_available[complete_mask].index
        outlier_indices = complete_indices[combined_outliers]

        # Replace outliers with NaN (only for available features);
        for feature in available_feature_cols:
            original_count = df_station[feature].isna().sum()
            df_station.loc[outlier_indices, feature] = np.nan
            new_count = df_station[feature].isna().sum()
            logger.info("%s: %s → %s missing values", feature, original_count, new_count)
            
            # Update status to track outlier replacement;
            status_col = feature + '_Status'
            if status_col in df_station.columns:
                df_station.loc[outlier_indices, status_col] = 5  # 5 = Outlier flagged;
        
        processed_stations.append(df_station)
    df_result = pd.concat(processed_stations, ignore_index=True)
    return df_result