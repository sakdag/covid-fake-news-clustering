import math
import nltk

import pandas as pd

from nltk import word_tokenize, pos_tag

import src.preprocessing.preprocessing as prep


def generate_raw_term_frequency_vectors(df: pd.DataFrame, raw_term_frequency_vectors_csv_file_name: str):
    term_set = prep.get_set_of_all_terms(df)

    # Dictionary to hold all raw term frequency values
    raw_term_freq_dict = dict()

    for term in raw_term_freq_dict.keys():
        raw_term_freq_dict[term] = list()

    for index, row in df.iterrows():
        current_headline = str(row['preprocessed_headlines'])

        for term in term_set:
            if term in raw_term_freq_dict.keys():
                raw_term_freq_dict[term].append(current_headline.count(term))
            else:
                raw_term_freq_dict[term] = [current_headline.count(term)]

    tf_idf_dataframe = pd.DataFrame(raw_term_freq_dict)
    tf_idf_dataframe.to_csv(raw_term_frequency_vectors_csv_file_name, index_label='docId')


def generate_tf_idf_vectors(df: pd.DataFrame, tf_idf_vectors_csv_file_name: str):
    idf_dict = generate_idf_map(df)

    # Dictionary to hold all tf-idf values of each term
    tf_idf_dict = dict()

    for term in idf_dict.keys():
        tf_idf_dict[term] = list()

    for index, row in df.iterrows():
        current_headline = str(row['preprocessed_headlines'])

        # If you want to apply maximum frequency normalization, comment out 2 lines below
        # and open the ones below that
        for term in idf_dict.keys():
            tf_idf_dict[term].append(current_headline.count(term) * idf_dict[term])

        # # Apply maximum frequency normalization
        # max_count = 0
        # for term in word_tokenize(current_headline):
        #     if current_headline.count(term) > max_count:
        #         max_count = current_headline.count(term)
        #
        # for term in idf_dict.keys():
        #     tf_idf_dict[term].append((0.5 + ((current_headline.count(term) * 0.5) / max_count)) * idf_dict[term])

    tf_idf_dataframe = pd.DataFrame(tf_idf_dict)
    tf_idf_dataframe.to_csv(tf_idf_vectors_csv_file_name, index_label='docId')


def generate_only_nouns_vectors(df: pd.DataFrame, tf_idf_only_nouns_vectors_file_name: str):
    # Modify dataset so that preprocessed_headlines holds only nouns
    # versions of the headlines, then utilize generate_tf_idf_vectors
    nltk.download('averaged_perceptron_tagger')

    modified_df = df.copy()
    for index, row in modified_df.iterrows():
        current_headline = str(row['preprocessed_headlines'])
        new_headline = ''

        pos_tagged = pos_tag(word_tokenize(current_headline))
        nouns = filter(lambda x: x[1] == 'NN', pos_tagged)

        for noun in nouns:
            new_headline += noun[0] + ' '

        if new_headline != '':
            new_headline = new_headline[:-1]

        modified_df.loc[index, 'preprocessed_headlines'] = new_headline

    generate_tf_idf_vectors(modified_df, tf_idf_only_nouns_vectors_file_name)


def generate_top_100_vectors(df: pd.DataFrame, tf_idf_top_100_vectors_file_name: str,
                             save_top_100: bool = False, save_file_name: str = 'Top100Words.txt'):
    idf_dict = generate_idf_map(df)
    global_tf_idf_dict = generate_global_tf_idf_dict(df, idf_dict)

    top_100_terms = []
    count = 0
    for element in sorted(global_tf_idf_dict.items(), key=lambda kv: kv[1], reverse=True):
        if count == 100:
            break
        top_100_terms.append(element[0])
        count += 1

    if save_top_100:
        f = open(save_file_name, "w")
        for term in top_100_terms:
            f.write(str(term) + ', ')
        f.close()

    modified_df = df.copy()
    for index, row in modified_df.iterrows():
        current_headline = str(row['preprocessed_headlines'])
        new_headline = ''

        for token in word_tokenize(current_headline):
            if token in top_100_terms:
                new_headline += token + ' '

        if new_headline != '':
            new_headline = new_headline[:-1]

        modified_df.loc[index, 'preprocessed_headlines'] = new_headline

    generate_tf_idf_vectors(modified_df, tf_idf_top_100_vectors_file_name)


def generate_global_tf_idf_dict(df: pd.DataFrame, idf_dict: dict):
    count_dict = dict()

    for index, row in df.iterrows():
        current_headline = str(row['preprocessed_headlines'])

        for term in word_tokenize(current_headline):
            if term in count_dict.keys():
                count_dict[term] += 1
            else:
                count_dict[term] = 1

    tf_idf_dict = dict()
    for key in sorted(idf_dict.keys()):
        tf_idf_dict[key] = count_dict[key] * idf_dict[key]

    return tf_idf_dict


def generate_idf_map(df: pd.DataFrame, term_idf_map_file_name: str = '', save: bool = False):
    idf_dict = dict()
    total_number_of_docs = len(df)

    # Count how many headlines each term appears in
    for index, row in df.iterrows():
        current_headline = row['preprocessed_headlines']
        token_list = word_tokenize(current_headline)

        # Remove duplicates
        token_list_wo_duplicates = list(dict.fromkeys(token_list))

        for token in token_list_wo_duplicates:
            if token in idf_dict.keys():
                idf_dict[token] += 1
            else:
                idf_dict[token] = 1

    # Compute IDF
    for key in sorted(idf_dict.keys()):
        idf_dict[key] = 1 + math.log(total_number_of_docs / int(idf_dict[key]), 10)

    if save:
        f = open(term_idf_map_file_name, "w")
        for key in sorted(idf_dict.keys()):
            f.write(str('"' + str(key) + '",' + str(idf_dict[key]) + '\n'))
        f.close()

    return idf_dict
