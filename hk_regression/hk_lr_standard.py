import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
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

pipeline_standard = make_pipeline(StandardScaler(), LinearRegression())
param_grid_standard = {
    'linearregression__fit_intercept': [True, False],
}
random_search_standard = RandomizedSearchCV(pipeline_standard, param_grid_standard, cv=5, n_jobs=-1, 
                                          scoring='neg_mean_squared_error', n_iter=2, random_state=42)

random_search_standard.fit(X_train, y_train)
y_pred_standard = random_search_standard.predict(X_test)

mse_standard = mean_squared_error(y_test, y_pred_standard)
mae_standard = mean_absolute_error(y_test, y_pred_standard)
rmse_standard = np.sqrt(mse_standard)
r2_standard = r2_score(y_test, y_pred_standard)

print("StandardScaler best params:", random_search_standard.best_params_)
print("StandardScaler MSE:", mse_standard)
print("StandardScaler MAE:", mae_standard)
print("Root Mean Squared Error (RMSE):", rmse_standard)
print("StandardScaler R^2 score:", r2_standard)
print("StandardScaler best score:", -random_search_standard.best_score_)