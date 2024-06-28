import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

df = pd.read_csv('data_to_ml.csv')

X = df.drop('Faction', axis=1)
y = df['Faction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_columns = ['Class', 'Rol', 'Class Type', 'Armor Type']
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

pipeline_standard = make_pipeline(StandardScaler(), LogisticRegression())
param_grid_standard = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
    'logisticregression__penalty': ['l2'],
    'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'logisticregression__max_iter': [1000, 2000, 3000],
    'logisticregression__class_weight': [None, 'balanced'],
}
grid_search_standard = GridSearchCV(pipeline_standard, param_grid_standard, cv=10, n_jobs=-1, verbose=1)

grid_search_standard.fit(X_train, y_train_encoded)
y_pred_standard = grid_search_standard.predict(X_test)
y_train_pred_standard = grid_search_standard.predict(X_train)
accuracy_standard = accuracy_score(y_test_encoded, y_pred_standard)

y_test_labels = label_encoder.inverse_transform(y_test_encoded)
y_pred_labels = label_encoder.inverse_transform(y_pred_standard)
y_train_labels = label_encoder.inverse_transform(y_train_encoded)
y_train_pred_labels = label_encoder.inverse_transform(y_train_pred_standard)

print("StandardScaler best params:", grid_search_standard.best_params_)
print("StandardScaler accuracy:", accuracy_standard)
print("StandardScaler best score:", grid_search_standard.best_score_)
print("Classification Report for Test Data")
print(classification_report(y_test_labels, y_pred_labels))
print("Classification Report for Training Data")
print(classification_report(y_train_labels, y_train_pred_labels))