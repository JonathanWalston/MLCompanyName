import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import os
import joblib

os.environ["LOKY_PICKLER"] = "pickle"

# Load the training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()  # Convert y_train to 1D array

# Compute sample weights based on the pattern_indicator
weights = X_train['pattern_indicator'].apply(lambda x: 10 if x == 1 else 1)

# Initialize a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

stratified_kfold = StratifiedKFold(n_splits=3)

# Use RandomizedSearchCV with 20 iterations
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=20,
                                   cv=stratified_kfold, scoring='accuracy', n_jobs=1, random_state=42)
random_search.fit(X_train, y_train, sample_weight=weights)

# Print the best hyperparameters found
print("Best Hyperparameters:", random_search.best_params_)

# Train the model using the best hyperparameters
best_rf = random_search.best_estimator_

# Save the trained model for future use
joblib.dump(best_rf, 'random_forest_model_updated.pkl')

