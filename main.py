import json
from itertools import cycle

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


def exclude_indexes(indexes, excluded_indexes):
    """
    Excludes sorted indexes from a sorted list.
    """
    result = []
    i = 0
    j = 0
    while i < len(indexes) and j < len(excluded_indexes):
        if indexes[i] < excluded_indexes[j]:
            result.append(indexes[i])
            i += 1
        elif indexes[i] > excluded_indexes[j]:
            j += 1
        else:
            i += 1
            j += 1
    if indexes[-1] > excluded_indexes[-1]:
        result.append(indexes[-1])
    return result


def multi_intersect_indexes(lists):
    """
    Intersects multiple sorted indexes.
    """
    if len(lists) == 1:
        return lists[0]

    result = []
    maxval = float("-inf")
    consecutive = 0
    try:
        for sublist in cycle(iter(sublist) for sublist in lists):

            value = next(sublist)
            while value < maxval:
                value = next(sublist)

            if value > maxval:
                maxval = value
                consecutive = 0
                continue

            consecutive += 1
            if consecutive >= len(lists) - 1:
                result.append(maxval)
                consecutive = 0

    except StopIteration:
        return result


def intersect_two_indexes(indexes1, indexes2):
    """
    Intersects two sorted indexes.
    """
    result = []
    i = 0
    j = 0
    while i < len(indexes1) and j < len(indexes2):
        if indexes1[i] == indexes2[j]:
            result.append(indexes1[i])
            i += 1
            j += 1
        elif indexes1[i] < indexes2[j]:
            i += 1
        else:
            j += 1
    return result


def multiple_word_query(words, word_index):
    """
    Answers to a multiple word query and sorts the results.
    """
    posting_lists = [word_index[word] for word in words]
    lists = [list(p['docs'].keys()) for p in posting_lists]
    result = multi_intersect_indexes(lists)
    ranked_result = np.zeros(len(result))
    for p in posting_lists:
        ranked_result += [p['docs'][i]['count'] for i in result]
    # return [x for x, y in sorted(zip(ranked_result, result), reverse=True)]
    return dict(zip(result, ranked_result))


def phrasal_query(phrasal_word, word_index):
    """
    Answers to a phrasal query and sorts the results.
    """
    words = phrasal_word.split()
    posting_lists = [word_index[word] for word in words]
    lists = [list(p['docs'].keys()) for p in posting_lists]
    intersect_of_words_in_phrase = multi_intersect_indexes(lists)

    result = {}
    for d in intersect_of_words_in_phrase:
        positions = [word_index[w]['docs'][d]['positions'] for w in words]
        for p in positions[0]:
            if all(p + i in positions[i] for i in range(1, len(positions))):
                if d in result:
                    result[d] += 1
                else:
                    result[d] = 1
    return result



def query(query, word_index, preprocessed=False):
    """
    Answers a query and sorts the results.
    supported operands: 1. double quotes("") for phrasal queries.
                        2. ! for negation.
                        3. otherwise intersects words.
    """
    if not preprocessed:
        query = preprocess_query(query)
    if len(query.split()) == 1:
        word = query.split()[0]
        if word not in word_index:
            return []
        posting_list = word_index[word]['docs']
        return sorted(posting_list, key=lambda x: posting_list[x]['count'], reverse=True)
    else:
        phrasal_words = re.findall(r'"(.*?)"', query)
        excluded_words = re.findall(r'!(.*?)!', query)
        other_words = re.sub(r'"(.*?)"|!(.*?)!', '', query).split()

        result = []
        if phrasal_words:
            result = multi_intersect_indexes([list(phrasal_query(phrasal_word, word_index).keys()) for phrasal_word in phrasal_words])
        if other_words:
            if phrasal_words:
                result = intersect_two_indexes(result, multiple_word_query(other_words, word_index).keys())
            else:
                result = list(multiple_word_query(other_words, word_index).keys())
        if excluded_words:
            for word in excluded_words:
                if word in word_index:
                    result = exclude_indexes(result, list(word_index[word]['docs'].keys()))
    return result


if __name__ == '__main__':
    # Load dataframe
    df = load_df('data/raw_data.json')
    # print(df.loc[6929])
    # Preprocess dataframe
    df = preprocess_df(df.loc, column_name='content', verbose=True)

    # Save dataframe
    df.csv('data/preprocessed_data.csv')

    # Create index dictionary
    word_index = create_index_dict(df, column_name='content')
    # json.dump(word_index, open('./data/word_index.json', 'w'))
    # with open('./data/word_index.json') as json_file:
    #     word_index = json.load(json_file)
    # # Single word query
    # print(query('فوتبال', word_index))
    # print(query('لیگ', word_index))
    #
    # # Multiple word query
    # print(query('بسکتبال لیگ', word_index))
    # print(query('فوتبال لیگ', word_index))
    #
    # # Excluded word query
    # print(query('!فوتبال! لیگ', word_index))
    # print(query('فوتبال !لیگ!', word_index))
    print(query('"تحریم هسته‌ای" آمریکا !ایران!', word_index))
