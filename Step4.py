import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())

# Save the trained model
joblib.dump(rf, 'random_forest_model.pkl')

# Load the dataset for prediction
data_to_predict = pd.read_csv('ML_ready_with_features.csv')

# Drop the columns not used for prediction
features_to_use = [
    'length_difference',
    'common_word_count',
    'shared_prefix_len',
    'jaccard_similarity',
    'fuzzy_token_set_ratio',
    '2gram_overlap',
    'pattern_indicator'
]
X_to_predict = data_to_predict[features_to_use]

# Predict using the model
predictions = rf.predict(X_to_predict)

# Add predictions to the dataframe
data_to_predict['Predicted Match'] = predictions

# Save the dataset with predictions
data_to_predict.to_csv('predicted_output.csv', index=False)
