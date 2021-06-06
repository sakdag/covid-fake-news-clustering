import math

import pandas as pd
from nltk import word_tokenize


def generate_tf_idf_vectors(df: pd.DataFrame, term_idf_map_file_name: str):
    idf_dict = generate_idf_map_and_save(df, term_idf_map_file_name)


def generate_idf_map_and_save(df: pd.DataFrame, term_idf_map_file_name: str):
    idf_dict = dict()
    total_number_of_docs = len(df)

    # Count how many headlines each term appears in
    for index, row in df.iterrows():
        current_headline = row['preprocessed_headlines']

        # Used to check if token is already considered for headline
        token_set = set()

        for token in word_tokenize(current_headline):
            if token not in token_set:
                if token in idf_dict.keys():
                    idf_dict[token] += 1
                else:
                    idf_dict[token] = 1
            token_set.add(token)

    f = open(term_idf_map_file_name, "w")

    # Compute IDF
    for key in sorted(idf_dict.keys()):
        idf_dict[key] = 1 + math.log(total_number_of_docs / idf_dict[key], 10)
        f.write(str(str(key) + ',' + str(idf_dict[key]) + '\n'))

    f.close()

    return idf_dict
