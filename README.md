# MLCompanyName

MLCompanyName is a Python-based project aimed at predicting company legal names based on their common names using machine learning techniques.

## Status

The scripts are currently optimized and tested for Windows. Other platforms have not been tested.

## Features

### step1.py
- Loads company and legal names from `company_and_legal.csv`.
- Uses fuzzy matching to generate potential matches between company names and legal names.
- Saves the results in `predicted_matches.csv`.

### step2.py
- Preprocesses the predicted matches by converting to lowercase and removing punctuation.
- Engineers features such as length difference, common word count, shared prefixes/suffixes, and more.
- Saves the dataset with engineered features in `ML_ready_with_features.csv`.

### step3.py
- Converts company names to strings for consistency.
- Splits the data into training and testing sets.
- Saves the training and testing data for further processing.

### step4.py
- Trains a Random Forest model using the training data.
- Predicts matches using the trained model.
- Saves the predictions in `predicted_output.csv`.

### step5.py
- Loads the trained Random Forest model.
- Evaluates the model on the test set and prints detailed classification metrics.
- If binary classification, plots the ROC curve.

### Step6_model_training.py
- Loads the training data and computes sample weights.
- Uses RandomizedSearchCV to optimize hyperparameters of the Random Forest model.
- Saves the trained model for future use.

### Step7_model_evaluation.py
- Loads the trained model and test data.
- Evaluates the model using various metrics.
- Uses the model to predict on a dataset and saves the refined predictions.

### Step8_model_interpretability.py
- Loads the trained model and a subset of the training data.
- Uses SHAP to explain the model's predictions and plots the SHAP values.

## Requirements

**Libraries:**
- pandas
- fuzzywuzzy
- sklearn
- shap
- matplotlib
- os
- joblib
- string

## Dataset

The `company_and_legal.csv` file contains a list of company names and their corresponding legal names. This dataset serves as the foundation for training and prediction tasks in this project.

## Quickstart

1. Clone or download the repository: `git clone https://github.com/JonathanWalston/MLCompanyName.git`
2. Install the required libraries using pip: pip install pandas fuzzywuzzy sklearn shap matplotlib joblib
3. Open the project in your preferred Python environment.
4. Run the Python scripts in the order from `step1.py` to `Step8_model_interpretability.py`.

## Note

The project was created with the assistance of generative AI tools for debugging and troubleshooting purposes.

## License

This project is open-source and free to use. Please ensure you follow the terms and conditions laid out in the LICENSE file.
