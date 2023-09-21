import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset with features
data = pd.read_csv('ML_ready_with_features.csv')

# Convert to strings to handle any non-string entries
data['Company Name'] = data['Company Name'].astype(str)
data['Predicted Legal Name'] = data['Predicted Legal Name'].astype(str)
data['Match'] = (data['Company Name'] == data['Predicted Legal Name']).astype(int)

# Features and target variable
features = [
    'length_difference', 'common_word_count', 'shared_prefix_len',
    'jaccard_similarity', 'fuzzy_token_set_ratio', '2gram_overlap', 'pattern_indicator'
]
X = data[features]
y = data['Match']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Save the training and testing data
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
