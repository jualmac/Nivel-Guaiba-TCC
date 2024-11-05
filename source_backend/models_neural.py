"""
Defines the NixtlaAutoModels class for automated time series forecasting using multiple models from the NeuralForecast 
library. This class facilitates the initialization with a dataset and model configurations. It supports external 
variables, identifies unique regions in the dataset, and processes data accordingly. The main functionality includes 
creating various forecasting models, generating predictions, and extracting optimal hyperparameters, all while 
maintaining modularity for ease of comparison between models;
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import json
import copy
import pandas as pd
from ray import tune
from typing import Tuple, List, Optional, Any, Dict, Type
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import RMSE
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import AutoTSMixer, AutoTSMixerx, AutoLSTM, AutoNBEATSx, AutoNHITS

########################################################################################################################
#
# MODEL
#
########################################################################################################################
class NixtlaAutoModels:
    def __init__(self, df: pd.DataFrame, forecasting_horizon: int, n_samples: int, freq: str = 'D', **kwargs):
        """
        Initializes the NixtlaAutoModels class with the provided dataset and parameters;

        Parameters:
            - df (pandas DataFrame): The input dataset containing time series data;
            - forecasting_horizon (int): The number of periods ahead to forecast;
            - freq (str, optional): The frequency of the time series ('D' for daily by default);
            - **kwargs: Additional model configuration options;
        """

        self.df = df
        self.forecasting_horizon = forecasting_horizon
        self.freq = freq
        self.kwargs = kwargs
        self.models = []
        self.n_samples = n_samples
        self.n_series = df.unique_id.nunique()
        self.val_size = int(0.10 * len(df.ds.unique()))
        self.test_size = int(0.20 * len(df.ds.unique()))
        self.external_variable_columns = self._get_external_variable_columns()

    def _get_external_variable_columns(self) -> list:
        """Determines the external variable columns"""
        non_external_variable_columns = ['ds', 'y', 'unique_id']
        all_columns = self.df.columns.to_list()
        return list(set(all_columns) - set(non_external_variable_columns))

    def _apply_external_variables(self, config: dict) -> None:
        """Applies external variables to the model configuration"""
        if config is not None:
            config['futr_exog_list'] = self.external_variable_columns

    def _create_models(self) -> None:
        """Creates and configures the forecasting models base on the configurations passed via **kwargs"""
        lstm_config = copy.deepcopy(self.kwargs.get('lstm_config', None))
        nhits_config = copy.deepcopy(self.kwargs.get('nhits_config', None))
        nbeatsx_config = copy.deepcopy(self.kwargs.get('nbeatsx_config', None))
        tsmixer_config = copy.deepcopy(self.kwargs.get('tsmixer_config', None))
        tsmixerx_config = copy.deepcopy(self.kwargs.get('tsmixerx_config', None))

        # Apply external variables to the relevant configs
        configs_with_external_data = [lstm_config, nhits_config, nbeatsx_config, tsmixerx_config]
        for config in configs_with_external_data:
            self._apply_external_variables(config)

        self.models = [
            AutoLSTM(
                h=self.forecasting_horizon, 
                num_samples=self.n_samples,
                config=lstm_config,
                search_alg=HyperOptSearch(), 
                loss=RMSE(), 
                valid_loss=RMSE(),
                verbose=True
            ),
            AutoNBEATSx(
                h=self.forecasting_horizon,
                loss=RMSE(),
                valid_loss=RMSE(),
                config=nbeatsx_config,
                num_samples=self.n_samples,
                search_alg=HyperOptSearch(),
                verbose=True
            ),
            AutoNHITS(
                h=self.forecasting_horizon,
                num_samples=self.n_samples, 
                config=nhits_config,
                search_alg=HyperOptSearch(), 
                loss=RMSE(), 
                valid_loss=RMSE(),
                verbose=True 
            ),
            AutoTSMixer(
                h=self.forecasting_horizon, 
                n_series=self.n_series,  
                config=tsmixer_config, 
                num_samples=self.n_samples, 
                search_alg=HyperOptSearch(), 
                loss=RMSE(),
                valid_loss=RMSE(),
                verbose=True
            ),
            AutoTSMixerx(
                h=self.forecasting_horizon, 
                n_series=self.n_series, 
                config=tsmixerx_config, 
                num_samples=self.n_samples, 
                search_alg=HyperOptSearch(), 
                loss=RMSE(), 
                valid_loss=RMSE(),
                verbose=True
            )                                        
        ]

    def _get_best_params(self, nf: NeuralForecast) -> list:
        """Extracts the best hyperparameters from the trained models"""
        best_param = []
        for model in nf.models:
            best_hyper = model.results.get_best_result().config
            serializable_dict = {
                key: (str(value) if callable(value) else value) 
                for key, value in best_hyper.items()
            }
            best_param.append(serializable_dict)

        with open('./source_backend/nixtla_params.json', 'w') as file:
            json.dump(best_param, file, indent=4)
        return best_param

    def last_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the last cutoff as the final result"""
        df = df.reset_index(drop=False)
        cutoffs = df['cutoff'].unique()
        last_cutoff = cutoffs[-1]

        window = df.loc[df.cutoff == last_cutoff]
        window.drop(['cutoff'], axis=1, inplace=True)
        return window

    def forecast(self) -> tuple:
        """
        Generates forecasts using the configured models;

        Returns:
            - tuple: Containing the last window predictions and all predictions;
        """
        self._create_models()
        nf = NeuralForecast(models=self.models, freq=self.freq)
        a = self.df
        y_pred = nf.cross_validation(df=self.df, val_size=self.val_size, test_size=self.test_size, n_windows=None)

        # Select only the last window as the final result;
        treated_predictions = self.last_window(y_pred)
        y_pred = y_pred.reset_index(drop=False)

        # Saving best hyperparameters:
        self._get_best_params(nf)
        return treated_predictions, y_pred