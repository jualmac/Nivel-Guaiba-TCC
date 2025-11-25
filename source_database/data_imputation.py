"""
Feature imputation operations;

Handles missing value imputation using multivariate iterative imputation with RandomForest;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
from typing import Tuple
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

########################################################################################################################
#                                                                  
# IMPUTATION FUNCTIONS
#
########################################################################################################################
def feature_imputation(
    df: pd.DataFrame, 
    n_estimators: int = 50
) -> Tuple[pd.DataFrame, dict]:
    """
    Impute missing values using IterativeImputer with RandomForestRegressor;
    
    Applies multivariate imputation per station to preserve within-station feature correlations.
    Uses IterativeImputer with RandomForestRegressor (max_depth=12, min_samples_leaf=10) for
    non-linear relationships. Skips fully missing columns. Excludes status columns and non-numeric
    features from imputation;
    
    Parameters:
        df (pd.DataFrame): Input dataframe with missing values to impute;
        n_estimators (int): Number of trees in RandomForestRegressor (default: 50);
    
    Returns:
        Tuple[pd.DataFrame, dict]: Tuple containing:
            - pd.DataFrame: Imputed dataframe with missing values filled;
            - dict: Per-station imputation statistics (initial_means, imputation_sequence, n_iter, cols_imputed, cols_skipped);
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    #TODO: Geographical Imputation for the Guaíba_1 (87450004) and Guaíba_2(87444000) Stations; Searched for Stations on
    # Rio_Codigo IN ('87200000') and did not found any station that is both Tipo_Estacao_Telemetrica IN ('1') and
    # Tipo_Rede_Classe_Vazao IN ('1') at the same time. There is a station very close that has Vazao, '87450005', but it
    # is not Tipo_Estacao_Telemetrica, and therefore the data can't be collected via the API; Also tried stations
    # '87500020' and '87460120'-> Flow data can't be used;
     
    # Separate index/categorical columns from numeric features;
    index_cols = ['Data_Hora_Medicao', 'codigoestacao']
    meta_cols = ['Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols + meta_cols
    feature_cols = list(set(df_cpy.columns.unique()) - set(non_feature_cols))
    
    # Impute per station to preserve within-station correlations;
    imputed_stations = []
    imputer_stats = {}
    
    for station in df_cpy['codigoestacao'].unique():
        print(f"\nImputing station {station}...")
        df_station = df_cpy[df_cpy['codigoestacao'] == station].copy()
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()
        
        # Identify fully missing columns for this station and exclude from imputation (Imputing 100% missing data would create synthetic values with no basis in reality);
        fully_missing = df_station_features.isna().all()
        cols_to_impute = [col for col in feature_cols if not fully_missing[col]]
        fully_missing_cols = [col for col in feature_cols if fully_missing[col]]
        
        if fully_missing_cols:
            print(f"  Skipping fully missing features (cannot impute): {fully_missing_cols}")
        
        if cols_to_impute:
            # Track which values were NaN before imputation (to mark status columns later);
            was_nan_before = df_station_features[cols_to_impute].isna()
            
            # RandomForestRegressor with deeper trees for better non-linear relationships (No max_depth restriction allows deep splits; min_samples_leaf=1 for fine-grained predictions);
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=12,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                random_state=42,
                max_iter=10,
                imputation_order='roman',
                verbose=1
            )
            
            # Impute only columns with partial data;
            imputed = imputer.fit_transform(df_station_features[cols_to_impute])
            df_station_imputed = pd.DataFrame(imputed, columns=cols_to_impute, index=df_station_features.index)
            
            # Mark status columns with code 4 (Filled/Missing) for imputed values;
            for col in cols_to_impute:
                status_col = col + '_Status'
                if status_col in df_station_non_features.columns:
                    is_now_filled = was_nan_before[col] & df_station_imputed[col].notna()
                    df_station_non_features.loc[is_now_filled, status_col] = 4
            
            # Store imputer statistics for diagnostics;
            imputer_stats[station] = {
                'initial_means': dict(zip(cols_to_impute, imputer.initial_imputer_.statistics_)),
                'imputation_sequence': imputer.imputation_sequence_,
                'n_iter': imputer.n_iter_,
                'cols_imputed': cols_to_impute,
                'cols_skipped': fully_missing_cols
            }
            
            # Keep fully missing columns as NaN (they will be excluded later in melt);
            for col in fully_missing_cols:
                df_station_imputed[col] = df_station_features[col]
        else:
            print(f"  Warning: No features to impute for station {station}")
            df_station_imputed = df_station_features
            imputer_stats[station] = {'error': 'No features to impute'}
        
        df_station_result = pd.concat([df_station_non_features, df_station_imputed], axis=1)
        imputed_stations.append(df_station_result)
    df_result = pd.concat(imputed_stations, ignore_index=True)
    
    # Convert status columns to integers to ensure they remain categorical;
    status_cols = [col for col in df_result.columns if col.endswith('_Status')]
    for status_col in status_cols:
        df_result[status_col] = df_result[status_col].astype('Int64')  # Nullable integer type;
    
    # Round only numeric feature columns;
    meta_cols = ['Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo']
    feature_cols = [col for col in df_result.columns if not col.endswith('_Status') and col not in ['Data_Hora_Medicao', 'codigoestacao'] + meta_cols]
    for col in feature_cols:
        df_result[col] = df_result[col].round(3)
    return df_result, imputer_stats