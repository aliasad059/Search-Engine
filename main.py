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


def preprocess_query(query):
    """
    Preprocesses a query.
    normalizes and tokenizes the query.
    removes stopwords, punctuation, urls ,and stemming.
    """
    query = re.sub(f'[{punctuation}؟،٪×÷»«]+', '', query)
    query = re.sub(r'http\S+', '', query)
    query = Normalizer().normalize(query)
    query = word_tokenize(query)
    query = [w for w in query if w not in stopwords_list()]
    query = [Stemmer().stem(w) for w in query]
    return ' '.join(query)


def preprocess_df(df, column_name, verbose=False):
    """
    Preprocesses a dataframe column.
    normalizes and tokenizes the text.
    removes stopwords, punctuation, urls ,and stemming.
    """
    if verbose:
        print('Removing URLs...')
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'http\S+', '', x))
    if verbose:
        print('Removing punctuations...')
    df[column_name] = df[column_name].apply(lambda x: re.sub(f'[{punctuation}؟،٪×÷»«]+', '', x))
    if verbose:
        print('Normalizing...')
    df[column_name] = df[column_name].apply(lambda x: Normalizer().normalize(x))
    if verbose:
        print('Tokenizing...')
    df[column_name] = df[column_name].apply(lambda x: word_tokenize(x))
    if verbose:
        print('Removing stopwords...')
    df[column_name] = df[column_name].apply(lambda x: [w for w in x if w not in stopwords_list()])
    if verbose:
        print('Stemming...')
    df[column_name] = df[column_name].apply(lambda x: [Stemmer().stem(w) for w in x])
    if verbose:
        print('Joining...')
    df[column_name] = df[column_name].apply(lambda x: ' '.join(x))
    if verbose:
        print('Done.')
    return df


def create_index_dict(df, column_name):
    """
    Creates a dictionary of word indexes.
    """
    word_index = {}
    for i, row in df.iterrows():
        for p, word in enumerate(row[column_name].split()):
            if word not in word_index:
                word_index[word] = {}
                word_index[word]['count'] = 1  # holds count of the word in all documents
                word_index[word]['docs'] = {}  # holds the documents that contain the word
                word_index[word]['docs'][i] = {}
                word_index[word]['docs'][i]['count'] = 1  # holds count of the word in document i
                word_index[word]['docs'][i]['positions'] = [p]  # holds the positions of the word in document i
            else:
                word_index[word]['count'] += 1
                if i not in word_index[word]['docs']:
                    word_index[word]['docs'][i] = {}
                    word_index[word]['docs'][i]['count'] = 1
                    word_index[word]['docs'][i]['positions'] = [p]
                else:
                    word_index[word]['docs'][i]['count'] += 1
                    word_index[word]['docs'][i]['positions'].append(p)
    return word_index


def query_to_index(query, word_index, preprocessed=False):
    """
    Converts a query to a list of word indexes.
    """
    if not preprocessed:
        query = preprocess_query(query)
    return [word_index[word] for word in query.split()]


def single_word_query(word, word_index):
    """
    Answers to a single word query and sorts the results.
    """
    if word not in word_index:
        return []

    posting_list = word_index[word]['docs']
    result = sorted(posting_list, key=lambda x: posting_list[x]['count'], reverse=True)
    return result


def multiple_word_query(query, word_index):
    """
    Answers to a multiple word query and sorts the results.
    """
    posting_lists = query_to_index(query, word_index)

    result = list(posting_lists[0]['docs'].keys())
    for posting_list in posting_lists[1:]:
        index = list(posting_list['docs'].keys())
        result = intersect_indexes(result, index)
    ranked_result = np.zeros(len(result))
    for p in posting_lists:
        ranked_result += [p['docs'][i]['count'] for i in result]
    return [x for y, x in sorted(zip(ranked_result, result), reverse=True)]


def intersect_indexes(index1, index2):
    """
    Intersects two sorted indexes.
    """
    result = []
    i = 0
    j = 0
    while i < len(index1) and j < len(index2):
        if index1[i] == index2[j]:
            result.append(index1[i])
            i += 1
            j += 1
        elif index1[i] < index2[j]:
            i += 1
        else:
            j += 1
    return result


if __name__ == '__main__':
    # Load dataframe
    df = load_df('data/raw_data.json')

    # Preprocess dataframe
    df = preprocess_df(df[0:5], column_name='content')

    # Save dataframe
    # df.to_json('data/preprocessed_data.json')

    # Create index dictionary
    word_index = create_index_dict(df, column_name='content')

    # Single word query
    print(single_word_query('بسکتبال', word_index))

    # # Multiple word query
    print(multiple_word_query('بسکتبال لیگ', word_index))
