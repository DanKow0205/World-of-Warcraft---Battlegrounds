import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import keras_tuner as kt

df = pd.read_csv('data_to_ml.csv')

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_columns = ['Faction', 'Rol', 'Class Type', 'Armor Type']
encoder = OneHotEncoder(handle_unknown='ignore')

X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))
X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

X_train.drop(categorical_columns, axis=1, inplace=True)
X_test.drop(categorical_columns, axis=1, inplace=True)

X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded_df], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded_df], axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)


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
    model.add(Dense(len(label_encoder.classes_), kernel_initializer='glorot_uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=hp.Choice('optimizer', values=['adam', 'SGD']),
                  metrics=['accuracy'])
    return model
    
tuner = kt.RandomSearch(build_model,
                        objective='val_accuracy',
                        max_trials=10,
                        executions_per_trial=3,
                        directory='keras_tuner_dir',
                        project_name='my_project')

tuner.search(X_train_scaled, y_train_categorical, epochs=500, validation_split=0.2)

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

history = model.fit(X_train_scaled, y_train_categorical, epochs=epochs, batch_size=batch_size, 
                    validation_split=0.2, callbacks=[early_stopping])

y_pred_test_encoded = model.predict(X_test_scaled)
y_pred_train_encoded = model.predict(X_train_scaled)

y_pred_test_num = np.argmax(y_pred_test_encoded, axis=1)
y_pred_train_num = np.argmax(y_pred_train_encoded, axis=1)

y_test_labels = label_encoder.inverse_transform(y_test_encoded)
y_pred_test_labels = label_encoder.inverse_transform(y_pred_test_num)
y_train_labels = label_encoder.inverse_transform(y_train_encoded)
y_pred_train_labels = label_encoder.inverse_transform(y_pred_train_num)

accuracy = accuracy_score(y_test_labels, y_pred_test_labels)

print("Best parameters found: ", best_hps.values)
print("Test accuracy: ", accuracy)
print("Classification Report for Test Data")
print(classification_report(y_test_labels, y_pred_test_labels, zero_division=0))
print("Classification Report for Training Data")
print(classification_report(y_train_labels, y_pred_train_labels, zero_division=0))