import pandas as pd
from src.pipelines.pipeline_data import pipeline_data
from src.pipelines.pipeline_preparation import encoding_pipeline
from src.models.train_models import training_pipeline

df = pipeline_data(save_to_db=False)
df = df.head(1000)
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

print("Processed X shape:", X_processed.shape)
print("Processed X columns:", X_processed.columns if hasattr(X_processed, 'columns') else "No columns attribute")
print("Processed X head:\n", X_processed.head(2) if hasattr(X_processed, 'head') else X_processed[:2])

import numpy as np
if hasattr(X_processed, 'select_dtypes'):
    X_num = X_processed.select_dtypes(include=[np.number])
    print("Numeric columns:", X_num.columns)
    print("Numeric shape:", X_num.shape)
