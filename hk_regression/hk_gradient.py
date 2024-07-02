import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv('data_to_ml.csv')

X = df.drop('HK', axis=1)
y = df['HK']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_columns = ['Faction', 'Class', 'Rol', 'Class Type', 'Armor Type']

encoder = OneHotEncoder(handle_unknown='ignore')

X_train_encoded = encoder.fit_transform(X_train[categorical_columns])

X_test_encoded = encoder.transform(X_test[categorical_columns])

X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))
X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

X_train.drop(categorical_columns, axis=1, inplace=True)
X_test.drop(categorical_columns, axis=1, inplace=True)

X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded_df], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded_df], axis=1)

param_grid_gradient_boosting = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 8, 64],     
    'max_features': [None, 'sqrt', 'log2'],
}

gradient_boosting = GradientBoostingRegressor()

random_search_gb = RandomizedSearchCV(gradient_boosting, param_grid_gradient_boosting, cv=5, n_jobs=-1, 
                                      scoring='neg_mean_squared_error', n_iter=50, random_state=42)

random_search_gb.fit(X_train, y_train)

y_pred_gb = random_search_gb.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("GradientBoostingRegressor best params:", random_search_gb.best_params_)
print("GradientBoostingRegressor MSE:", mse_gb)
print("GradientBoostingRegressor MAE:", mae_gb)
print("Root Mean Squared Error (RMSE):", rmse_gb)
print("GradientBoostingRegressor R^2 score:", r2_gb)
print("GradientBoostingRegressor best score:", random_search_gb.best_score_)