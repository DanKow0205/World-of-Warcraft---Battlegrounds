import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from sklearn.ensemble import GradientBoostingClassifier

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

param_grid_gradient_boosting = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 8, 64],     
    'max_features': [None, 'sqrt', 'log2'],
}

gradient_boosting = GradientBoostingClassifier()

grid_search = GridSearchCV(gradient_boosting, param_grid_gradient_boosting, cv=10, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train_encoded)

y_pred_test = grid_search.predict(X_test)
y_pred_train = grid_search.predict(X_train)

y_test_labels = label_encoder.inverse_transform(y_test_encoded)
y_pred_test_labels = label_encoder.inverse_transform(y_pred_test)
y_train_labels = label_encoder.inverse_transform(y_train_encoded)
y_pred_train_labels = label_encoder.inverse_transform(y_pred_train)

accuracy = accuracy_score(y_test_labels, y_pred_test_labels)

print("GradientBoostingClassifier best params:", grid_search.best_params_)
print("GradientBoostingClassifier accuracy:", accuracy)
print("GradientBoostingClassifier best score:", grid_search.best_score_)
print("Classification Report for Test Data")
print(classification_report(y_test_labels, y_pred_test_labels))
print("Classification Report for Training Data")
print(classification_report(y_train_labels, y_pred_train_labels))