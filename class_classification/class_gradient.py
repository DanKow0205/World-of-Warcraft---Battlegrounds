import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
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

param_grid_gradient_boosting = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 8, 64],     
    'max_features': [None, 'sqrt', 'log2'],
}

gradient_boosting = GradientBoostingClassifier()

random_search_gb = RandomizedSearchCV(gradient_boosting, param_grid_gradient_boosting, n_iter=50, cv=5, n_jobs=-1,
                                   verbose=1, random_state=42)

random_search_gb.fit(X_train, y_train_encoded)

y_pred_test_gb = random_search_gb.predict(X_test)
y_pred_train_gb = random_search_gb.predict(X_train)

y_test_labels_gb = label_encoder.inverse_transform(y_test_encoded)
y_pred_test_labels_gb = label_encoder.inverse_transform(y_pred_test_gb)
y_train_labels_gb = label_encoder.inverse_transform(y_train_encoded)
y_pred_train_labels_gb = label_encoder.inverse_transform(y_pred_train_gb)

accuracy_gb = accuracy_score(y_test_labels_gb, y_pred_test_labels_gb)

print("GradientBoostingClassifier best params:", random_search_gb.best_params_)
print("GradientBoostingClassifier accuracy:", accuracy_gb)
print("GradientBoostingClassifier best score:", random_search_gb.best_score_)
print("Classification Report for Test Data")
print(classification_report(y_test_labels_gb, y_pred_test_labels_gb))
print("Classification Report for Training Data")
print(classification_report(y_train_labels_gb, y_pred_train_labels_gb))