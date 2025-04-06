# llm_label_filtering_pipeline.py
# Scalable Labeling Pipeline using LLMs + Filtering Rules (for QA, Cleanup, or Auto-Verification)

import openai  # or use HuggingFace LLMs via transformers
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
openai.api_key = "YOUR_OPENAI_API_KEY"
LABELING_GUIDELINES = """
Assign one of the following labels based on the sentence:
[positive, neutral, negative]
- Positive: clear positive sentiment or praise.
- Neutral: objective or factual.
- Negative: dissatisfaction, complaint, or criticism.
"""

# --- RULE-BASED FILTERS ---
def rule_based_flag(text):
    flags = []
    if len(text.split()) < 3:
        flags.append("too_short")
    if re.search(r'\b(?:lorem|ipsum|asdf)\b', text.lower()):
        flags.append("placeholder_text")
    if re.search(r'(.)\1{3,}', text):
        flags.append("repetition")
    return flags

# --- LLM-BASED FILTERING ---
def query_llm_label(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": LABELING_GUIDELINES},
            {"role": "user", "content": f"Text: {text}\nLabel:"}
        ]
    )
    return response['choices'][0]['message']['content'].strip().lower()

# --- SEMANTIC SIMILARITY FILTERING ---
def flag_duplicates(df, threshold=0.9):
    vectorizer = TfidfVectorizer().fit_transform(df['text'])
    sims = cosine_similarity(vectorizer)
    duplicate_pairs = set()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sims[i, j] > threshold:
                duplicate_pairs.add((i, j))
    return list(duplicate_pairs)

# --- PIPELINE ---
def run_pipeline(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['flags'] = df['text'].apply(rule_based_flag)
    df['llm_label'] = df['text'].apply(query_llm_label)

    duplicates = flag_duplicates(df)
    for i, j in duplicates:
        df.at[i, 'flags'].append("duplicate_with_{}".format(j))

    df.to_csv(output_csv, index=False)
    print(f"Filtered and labeled data saved to {output_csv}")

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    run_pipeline("raw_user_feedback.csv", "cleaned_labeled_feedback.csv")
