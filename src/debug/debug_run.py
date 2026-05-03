import pandas as pd
from src.pipelines.pipeline_data import pipeline_data
from src.pipelines.pipeline_preparation import encoding_pipeline
from src.models.train_models import training_pipeline
from src.models.model_lstm import LSTMModels

df = pipeline_data(save_to_db=False).head(100)
X = df.drop(columns=['Cota_Adotada_87450004'])
y = df['Cota_Adotada_87450004']

preprocessor = encoding_pipeline(target_column='Cota_Adotada_87450004')
preprocessor.fit(X, y)

pipelines = training_pipeline(
    preprocessor=preprocessor,
    models_to_use=['LSTM'],
    use_lags=True,
    use_rolling_stats=True,
    use_cumulative=True,
    use_feature_selection=True,
    n_features=50,
    log_mlflow=False
)

pipe = pipelines['LSTM']
feature_steps = pipe.steps[:-1]
from sklearn.pipeline import Pipeline
feature_pipeline = Pipeline(feature_steps)

feature_pipeline.fit(X, y)
X_processed = feature_pipeline.transform(X)

model_step_name, model_instance = pipe.steps[-1]
model_params = {
    "optimize_hyperparameters": False,
    "X_val": X_processed,
    "y_val": y,
    "early_stopping": 1,
    "feature_pipeline": feature_pipeline,
    "X_raw": X,
    "cv_n_splits": 2,
    "cv_gap": 24,
    "epochs": 2
}
model_instance.fit(X_processed, y, **model_params)
print("SUCCESS!")
