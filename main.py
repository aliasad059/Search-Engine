from hazm import Normalizer, word_tokenize, stopwords_list, Stemmer
import numpy as np
import pandas as pd
import re
from string import punctuation


def load_df(file_name):
    """
    Loads a dataframe from a json file.
    """
    return pd.read_json(file_name).transpose()


def preprocess_df(df, verbose=False):
    """
    Preprocesses a dataframe.
    normalizes and tokenizes the text.
    removes stopwords, punctuation, urls ,and stemming.
    """
    if verbose:
        print('Removing punctuations...')
    df['content'] = df['content'].apply(lambda x: re.sub(f'[{punctuation}؟،٪×÷»«]+', '', x))
    if verbose:
        print('Removing URLs...')
    df['content'] = df['content'].apply(lambda x: re.sub(r'http\S+', '', x))
    if verbose:
        print('Normalizing...')
    df['content'] = df['content'].apply(lambda x: Normalizer().normalize(x))
    if verbose:
        print('Tokenizing...')
    df['content'] = df['content'].apply(lambda x: word_tokenize(x))
    if verbose:
        print('Removing stopwords...')
    df['content'] = df['content'].apply(lambda x: [w for w in x if w not in stopwords_list()])
    if verbose:
        print('Stemming...')
    df['content'] = df['content'].apply(lambda x: [Stemmer().stem(w) for w in x])
    if verbose:
        print('Joining...')
    df['content'] = df['content'].apply(lambda x: ' '.join(x))
    if verbose:
        print('Done.')
    return df


if __name__ == '__main__':
    # Load dataframe
    df = load_df('data/raw_data.json')

    # Preprocess dataframe
    df = preprocess_df(df[0:5])

    # Save dataframe
    # df.to_json('data/preprocessed_data.json')
