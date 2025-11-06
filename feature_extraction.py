import pandas as pd
import string
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
POS_WHITELIST = {"VERB", "NOUN", "ADV"}

def pos_counts(text):
    doc = nlp(text)
    return Counter(token.pos_ for token in doc if token.pos_ in POS_WHITELIST)

def add_pos_features(df):
    pos_data = df["cleaned_text"].apply(pos_counts)
    pos_df = pd.DataFrame(list(pos_data)).fillna(0)
    pos_df.index = df.index
    return pd.concat([df, pos_df], axis=1)

def extract_features(df, include_pos=False):
    df["char_length"] = df["cleaned_text"].apply(len)
    df["word_count"] = df["cleaned_text"].str.split().apply(len)
    df["punctuation_ct"] = df["cleaned_text"].apply(lambda x: sum(1 for c in x if c in string.punctuation))
    df["is_extreme_star"] = df["rating"].isin([1.0, 5.0])
    if include_pos:
        df = add_pos_features(df)
    return df