import pandas as pd
from fuzzywuzzy import fuzz, process

# Load the data
df = pd.read_csv('company_and_legal.csv')

# Extract unique company names and legal names
company_names = df['Company Name'].unique()
legal_names = df['Legal Name'].unique()

# Use fuzzy matching to generate potential matches
top_matches = {}
for company in company_names:
    match, score = process.extractOne(company, legal_names, scorer=fuzz.token_set_ratio)
    if score > 50:  # Using a threshold to filter out very weak matches
        top_matches[company] = match

# Convert the dictionary to a DataFrame
matched_df = pd.DataFrame(list(top_matches.items()), columns=['Company Name', 'Predicted Legal Name'])

# Save the results
matched_df.to_csv('predicted_matches.csv', index=False)
