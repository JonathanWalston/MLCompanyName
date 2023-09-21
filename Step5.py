import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Predict on the test set
y_pred = model.predict(X_test)

# Print classification report for detailed metrics
print(classification_report(y_test, y_pred))

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# If it's a binary classification, you can also print the ROC AUC score
if len(y_test['Match'].unique()) == 2:
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
