import string

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords


def read_dataset(path):
    df = pd.read_csv(path)
    df_size = len(df)
    print('Data Size: ' + str(df_size))
    return df


def preprocess_and_save(df: pd.DataFrame, preprocessed_file_name: str):
    nltk.download('stopwords')
    stop = stopwords.words('english')

    df['preprocessed_headlines'] = np.NaN

    for index, row in df.iterrows():
        current_headline = row['headlines']
        preprocessed_headline = ''

        # For each term remove stopwords, punctuation symbols.
        # Also change token to lowercase letter version.
        for term in word_tokenize(current_headline):
            term = term.lower()
            if term not in stop and term not in string.punctuation:
                preprocessed_headline += term + ' '

        df.loc[index, 'preprocessed_headlines'] = preprocessed_headline[:-1]

    df.drop(columns=['headlines'], inplace=True)

    df.to_csv(preprocessed_file_name, index_label='docId')


def get_set_of_all_terms(df: pd.DataFrame):
    term_set = set()

    for index, row in df.iterrows():
        current_headline = str(row['preprocessed_headlines'])

        for term in word_tokenize(current_headline):
            term_set.add(term)

    return term_set
