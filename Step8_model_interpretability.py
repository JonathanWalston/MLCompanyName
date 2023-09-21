import pandas as pd
import joblib
import shap

# Load the trained model
model = joblib.load('random_forest_model_updated.pkl')


# Load the training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Use a subset of the training data for interpretability (for faster computation)
X_sample = X_train.sample(n=500, random_state=42)

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Plot the SHAP values
shap.summary_plot(shap_values[1], X_sample)
