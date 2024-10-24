"""
Dictionaries with the Nixtla configurations for the General Regression. These dictionaries 
They are present here so the main.py code 
doesn't get too clustered. This is defined as a .py file and not as a .json or .yml file because of the usage of the 
ray tune functions. Using these respective filetypes could introduce security issues on the code. Contains the 
configurations for the following models:

- Long short-term memory (LSTM);
- Neural hierarchical interpolation for Time Series forecasting (NHITS);
- Neural basis expansion analysis for interpretable time series forecasting w/ exogenous variables (NBEATSX);
- Time-Series Mixer (TSMIXER);
- Time-Series Mixer w/ exogenous variables (TSMIXERX);
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import copy
from ray import tune
from neuralforecast.auto import AutoNBEATSx

########################################################################################################################
#                                                                  
# DICTIONARIES
#
########################################################################################################################
prediction_horizon = 6
early_stop_steps = 5
random_seed = 42

lstm_config = {
    "input_size": tune.choice([30, 35, 40]),                      # Maximum sequence length
    "encoder_hidden_size": tune.choice([50, 60, 70]),                     # Hidden size of LSTM cells
    "encoder_n_layers": tune.choice([4]),                      # Number of layers in LSTM
    "encoder_dropout": tune.choice([0.1, 0.5]),                       # Dropout regularization applied to LSTM outputs
    "decoder_hidden_size": tune.choice([150, 200, 250]),
    "decoder_layers": tune.choice([4]),   
    "learning_rate": tune.loguniform(1e-4, 1e-3),                 # Initial Learning rate
    "scaler_type": tune.choice(['robust', 'standard']),                       # Scaler type
    "max_steps": tune.choice([250, 500]),                         # Max number of training iterations
    "batch_size": 2**5,                                           # Number of series in batch
    "random_seed": random_seed, 
    "early_stop_patience_steps": early_stop_steps,
}

nhits_config = {
    "input_size": tune.choice([4, 8, 16, 32]),                      # Maximum sequence length
    "start_padding_enabled": True,
    "n_blocks": 5*[1],                                              # Length of input window
    "mlp_units": 5 * [[64, 64]],                                    # Length of input window
    "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],         
                                    [8, 4, 2, 1, 1]]),              # MaxPooling Kernel size
    "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
                                    [1, 1, 1, 1, 1]]),              # Interpolation expressivity ratios
    "learning_rate": tune.loguniform(1e-4, 1e-3),                   # Initial Learning rate
    "scaler_type": tune.choice(['robust', 'standard']),                             # Scaler type
    "max_steps": tune.choice([250, 500]),                           # Max number of training iterations      
    "windows_batch_size": tune.choice([128, 256, 512]),             # Number of windows in batch
    "batch_size": 2**7,                                             # Number of series in batch
    "random_seed": random_seed, 
    "early_stop_patience_steps": early_stop_steps,
}

nbeatsx_config = AutoNBEATSx.get_default_config(h = prediction_horizon, backend="ray")
nbeatsx_config['early_stop_patience_steps'] = early_stop_steps
nbeatsx_config['random_seed'] = 42

tsmixer_config = {
    "input_size": tune.choice([4, 8, 16, 32]),          # Size of input window
    "max_steps": tune.choice([200, 500, 700]),         # Number of training iterations
    "val_check_steps": 100,                             # Compute validation every x steps
    "learning_rate": tune.loguniform(1e-5, 1e-3),       # Initial Learning rate
    "n_block": tune.choice([1, 2, 4, 6, 8]),            # Number of mixing layers
    "dropout": tune.uniform(0.0, 0.99),                 # Dropout
    "ff_dim": tune.choice([32, 64, 128]),               # Dimension of the feature linear layer
    "scaler_type": tune.choice(['identity', 'robust']),
    "batch_size": 2**6,                                 # Number of series in batch       
    "random_seed": random_seed, 
    "early_stop_patience_steps": early_stop_steps,      # Early stopping steps
    }

tsmixerx_config = tsmixer_config.copy()