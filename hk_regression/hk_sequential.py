import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping

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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units=hp.Int('neurons', min_value=1, max_value=10, step=1),
                    kernel_initializer=hp.Choice('init', values=['glorot_uniform', 'he_normal']),
                    activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.2, step=0.1)))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=hp.Choice('optimizer', values=['adam', 'SGD']),
                  metrics=['mean_squared_error'])
    return model

tuner = kt.RandomSearch(build_model,
                        objective='val_mean_squared_error',
                        max_trials=10,
                        executions_per_trial=3,
                        directory='keras_tuner_dir',
                        project_name='my_regression_project')

tuner.search(X_train_scaled, y_train, epochs=500, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = build_model(best_hps)

epochs = best_hps.get('epochs') if 'epochs' in best_hps.values else 500
batch_size = best_hps.get('batch_size') if 'batch_size' in best_hps.values else 10

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

y_pred = model.predict(X_test_scaled)

mse_s = mean_squared_error(y_test, y_pred)
mae_s = mean_absolute_error(y_test, y_pred)
rmse_s = np.sqrt(mse_s)
r2_s = r2_score(y_test, y_pred)

print("Best parameters found: ", best_hps.values)
print("Mean Squared Error (MSE): ", mse_s)
print("Mean Absolute Error (MAE): ", mae_s)
print("Root Mean Squared Error (RMSE): ", rmse_s)
print("R^2 score: ", r2_s)