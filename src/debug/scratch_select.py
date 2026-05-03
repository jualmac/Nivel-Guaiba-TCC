import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

X = np.random.rand(100, 10)
y = np.random.rand(100)

model = XGBRegressor(n_estimators=10, max_depth=3)
model.fit(X, y)

model.feature_importances_[:] = 0 # Force 0 importances to simulate this condition

selector = SelectFromModel(model, prefit=True, max_features=50, threshold=-np.inf)
X_trans = selector.transform(X)
print("X_trans shape:", X_trans.shape)

# Let's also check what XGBRegressor actual importances look like on purely random data
model2 = XGBRegressor(n_estimators=10, max_depth=3)
model2.fit(X, y)
print("Importances:", model2.feature_importances_)
selector2 = SelectFromModel(model2, prefit=True, max_features=50, threshold=-np.inf)
print("Selector2 shape:", selector2.transform(X).shape)
