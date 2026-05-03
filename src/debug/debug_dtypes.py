import pandas as pd
from src.pipelines.pipeline_data import pipeline_data
from src.pipelines.pipeline_preparation import encoding_pipeline
from src.models.train_models import training_pipeline

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

feature_pipeline = __import__('sklearn.pipeline').pipeline.Pipeline(pipelines['LSTM'].steps[:-1])
feature_pipeline.fit(X, y)
X_processed = feature_pipeline.transform(X)

print(X_processed.dtypes.value_counts())
