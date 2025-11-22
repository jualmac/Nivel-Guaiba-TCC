"""
Main code that concatenates other functions defined in the module. This will read the data from the database, 
clean it, add exeternal data, train the models and calculate the errors for the Regression;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import argparse
import pandas as pd
from ray import tune
from source_backend.data_transformation import get_data
from source_backend.data_handler import clean, add_external_data
from source_backend.models_neural import NixtlaAutoModels
from source_backend.models_regression import run_all_regression_models
from source_backend.calculate_errors import errors
from source_backend.best_model import best_model
from source_backend.config_nixtla import lstm_config, nhits_config, nbeatsx_config, tsmixer_config, tsmixerx_config

########################################################################################################################
#                                                                  
# QUERY AND DEFINITIONS
#
########################################################################################################################
def main(args) -> None:
    # Argparse variables;
    prediction_horizon = args.pred
    data_freq = args.freq
    batch_size = args.batch
    n_samples = args.samples

    print(f'Selected variables: {args}')

    # Change the batch_size for the defined Nixtla configurations;
    configs = [
        lstm_config, 
        nhits_config, 
        nbeatsx_config, 
        tsmixer_config, 
        tsmixerx_config
        ]

    for config in configs:
        config["batch_size"] = batch_size

    # Run models for each reagion and merge the results in a single df;
    for region in regions:
        df = data_ext.loc[data_ext['region'] == region]
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Train and Test Maximum Division. 'region' has to be droped because of the Regression Models; 
        df_test = df.sort_index().iloc[-prediction_horizon:].drop(['region'], axis=1)
        df_train = df.sort_index().iloc[:-prediction_horizon].drop(['region'], axis=1)

        X_train, y_train = df_train.iloc[:, 1:], df_train['target']
        X_test, y_test = df_test.iloc[:, 1:], df_test['target']

        # Ensure the frequency is set;
        X_train, y_train, X_test, y_test = [dataset.asfreq(data_freq) for dataset in [X_train, y_train, X_test, y_test]]
        
        # Train regression models for the General Prediction;
        data_regression = run_all_regression_models(X_train=X_train, y_train=y_train, X_test=X_test)
        data_regression['region'] = region
        results = pd.concat([results, data_regression], axis=0)

    # Prepare dataframe to be compatible with the Nixtla Library;
    data_ext = data_ext.reset_index(drop=False) #The timestamp should be a atribute, not the index;
    data_ext.rename(columns={"date": "ds", "target": "y", "region": "unique_id"}, inplace=True)
    data_ext['ds'] = pd.to_datetime(data_ext['ds'])

    # Run Neural Nixtla Models for the General Prediction;
    model_data = NixtlaAutoModels(
        df=data_ext, 
        forecasting_horizon=prediction_horizon,
        n_samples=5,
        freq=data_freq, 
        lstm_config=lstm_config_data,
        nhits_config=nhits_config_data, 
        nbeatsx_config=nbeatsx_config_data, 
        tsmixer_config=tsmixer_config_data, 
        tsmixerx_config=tsmixerx_config_data
        )

    data_nixtla, data_nixtla_windows = model_data.forecast()
    data_nixtla.rename(columns={"ds": "date", "unique_id": "region"}, inplace=True)

    # Merge results;
    data_results = pd.merge(results, data_nixtla, on=['date', 'region']) #TODO Change columns order;

    # Calculate Errors;
    data_errors = errors_data(data_results)

    # Calculate Best Model;
    data_best = best_data(data_results, data_errors)
    print('DONE')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data from a specified HANA view in a given schema.')
    parser.add_argument('--pred', type=int, default=6, help='The amount of fowards steps to be predicted')
    parser.add_argument('--freq', type=str, choices=['h', 'bh', 'min', 's', 'D', 'B', 'W', 'M', 'MS', 'SMS'], default='MS', help='Frequency of predictions (pandas offset)')
    parser.add_argument('--batch', type=int, default=64, help='Training batch size')
    parser.add_argument('--samples', type=int, default=2, help='Number of samples for the Nixtla models fine tunning')
    args = parser.parse_args()
    
    main(args)