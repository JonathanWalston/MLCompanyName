import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Load the trained model
rf_model = joblib.load('random_forest_model.pkl')

# Load the test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()  # Convert y_test to 1D array

# Make predictions
y_pred = rf_model.predict(X_test)

# Post-processing: Override predictions based on the pattern_indicator
y_pred[X_test['pattern_indicator'] == 1] = 1

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print a detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Non-Match', 'Match']))

# Plot ROC curve
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Load the dataset for prediction
data_to_predict = pd.read_csv('ML_ready_with_features.csv')

# Extract the features for prediction
X_to_predict = data_to_predict[[
    'length_difference',
    'common_word_count',
    'shared_prefix_len',
    'jaccard_similarity',
    'fuzzy_token_set_ratio',
    '2gram_overlap',
    'pattern_indicator'
]]

# Predict using the refined model
predictions = rf_model.predict(X_to_predict)

# Post-processing: Override predictions based on the pattern_indicator
predictions[X_to_predict['pattern_indicator'] == 1] = 1

# Add predictions to the dataframe
data_to_predict['Predicted Match'] = predictions

# Save the dataset with refined predictions
data_to_predict.to_csv('predicted_output_updated.csv', index=False)
