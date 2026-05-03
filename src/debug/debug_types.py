import pandas as pd
import numpy as np

# Let's recreate FeatureImportanceSelector transform logic
X_numeric = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]}, index=[0, 1])
X_numeric_array = X_numeric.to_numpy()
print("X_numeric_array dtype:", X_numeric_array.dtype)

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

y = np.array([1, 2])
model = XGBRegressor(n_estimators=10)
model.fit(X_numeric_array, y)

selector = SelectFromModel(model, prefit=True, max_features=1, threshold=-np.inf)
X_trans = selector.transform(X_numeric_array)
print("X_trans dtype:", X_trans.dtype)

df_trans = pd.DataFrame(X_trans, columns=['A'])
print("df_trans dtypes:")
print(df_trans.dtypes)
