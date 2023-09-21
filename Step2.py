import pandas as pd
from fuzzywuzzy import fuzz
import string

# Load the dataset
predicted_matches_df = pd.read_csv('predicted_matches.csv')

# Convert to strings to handle any non-string entries and preprocess
predicted_matches_df['Company Name'] = predicted_matches_df['Company Name'].str.lower().astype(str).str.translate(str.maketrans('', '', string.punctuation))
predicted_matches_df['Predicted Legal Name'] = predicted_matches_df['Predicted Legal Name'].str.lower().astype(str).str.translate(str.maketrans('', '', string.punctuation))

# Feature Engineering

# 1. String Length Differences
predicted_matches_df['length_difference'] = abs(predicted_matches_df['Company Name'].str.len() - predicted_matches_df['Predicted Legal Name'].str.len())

# 2. Common Word Counts
predicted_matches_df['common_word_count'] = predicted_matches_df.apply(lambda row: len(set(row['Company Name'].split()) & set(row['Predicted Legal Name'].split())), axis=1)


# 3. Shared Prefixes/Suffixes
def common_prefix_length(str1, str2):
    common_prefix = 0
    min_length = min(len(str1), len(str2))
    for i in range(min_length):
        if str1[i] == str2[i]:
            common_prefix += 1
        else:
            break
    return common_prefix


predicted_matches_df['shared_prefix_len'] = predicted_matches_df.apply(lambda row: common_prefix_length(row['Company Name'], row['Predicted Legal Name']), axis=1)


# 4. Token-based Similarities (Jaccard)
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return len(set1 & set2) / float(len(set1 | set2))


predicted_matches_df['jaccard_similarity'] = predicted_matches_df.apply(lambda row: jaccard_similarity(row['Company Name'], row['Predicted Legal Name']), axis=1)


# 5. Fuzzy Matching Scores
predicted_matches_df['fuzzy_token_set_ratio'] = predicted_matches_df.apply(lambda row: fuzz.token_set_ratio(row['Company Name'], row['Predicted Legal Name']), axis=1)


# 6. N-gram Overlaps (2-grams)
def ngram_overlap(str1, str2, n=2):
    ngrams_str1 = set(["".join(j) for j in zip(*[str1[i:] for i in range(n)])])
    ngrams_str2 = set(["".join(j) for j in zip(*[str2[i:] for i in range(n)])])
    return len(ngrams_str1 & ngrams_str2)


predicted_matches_df['2gram_overlap'] = predicted_matches_df.apply(lambda row: ngram_overlap(row['Company Name'], row['Predicted Legal Name']), axis=1)


# 7. Pattern Indicator based on identified patterns
def pattern_indicator(row):
    if row['shared_prefix_len'] > 7 and row['2gram_overlap'] > 7 and row['jaccard_similarity'] > 0.33332 and row['fuzzy_token_set_ratio'] > 87:
        return 1
    else:
        return 0


predicted_matches_df['pattern_indicator'] = predicted_matches_df.apply(pattern_indicator, axis=1)

# Save the dataset with features
predicted_matches_df.to_csv('ML_ready_with_features.csv', index=False)
