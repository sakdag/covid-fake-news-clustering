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


def read_dataset_with_index(path, index_column):
    df = pd.read_csv(path, index_col=index_column)
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

        for term in word_tokenize(current_headline):
            # Lowercase the token
            term = term.lower()

            # Remove non printable characters
            characters_to_hold = set(string.ascii_letters + string.digits)
            for character in term:
                if character not in characters_to_hold:
                    term = term.replace(character, "")

            # Remove hyphens and apostrophes from terms
            # term = term.replace("'", "")
            # term = term.replace("-", "")

            # Remove single word terms and words that are stopwords
            if len(term) < 2 or term in stop:
                continue

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
