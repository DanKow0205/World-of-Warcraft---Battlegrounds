import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical

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

pipeline_minmax = make_pipeline(MinMaxScaler(), LogisticRegression())
param_grid_minmax = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
    'logisticregression__penalty': ['l2'],
    'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'logisticregression__max_iter': [1000, 2000, 3000],
    'logisticregression__class_weight': [None, 'balanced'],
}
random_search_minmax = RandomizedSearchCV(pipeline_minmax, param_grid_minmax, n_iter=50, cv=5, n_jobs=-1, 
                                          verbose=1, random_state=42)

random_search_minmax.fit(X_train, y_train_encoded)
y_pred_test_minmax = random_search_minmax.predict(X_test)
y_pred_train_minmax = random_search_minmax.predict(X_train)

y_test_labels_minmax = label_encoder.inverse_transform(y_test_encoded)
y_pred_test_labels_minmax = label_encoder.inverse_transform(y_pred_test_minmax)
y_train_labels_minmax = label_encoder.inverse_transform(y_train_encoded)
y_pred_train_labels_minmax = label_encoder.inverse_transform(y_pred_train_minmax)

accuracy_minmax = accuracy_score(y_test_labels_minmax, y_pred_test_labels_minmax)

print("MinMaxScaler best params:", random_search_minmax.best_params_)
print("MinMaxScaler accuracy:", accuracy_minmax)
print("MinMaxScaler best score:", random_search_minmax.best_score_)
print("Classification Report for Test Data")
print(classification_report(y_test_labels_minmax, y_pred_test_labels_minmax))
print("Classification Report for Training Data")
print(classification_report(y_train_labels_minmax, y_pred_train_labels_minmax))