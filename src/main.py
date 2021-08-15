import argparse
import os

import src.config.config as conf
import src.preprocessing.preprocessing as prep
import src.preprocessing.vector_generation as vg
import src.clustering.clustering as clustering

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    news_dataset_file_path = os.path.join(dirname, conf.FAKE_NEWS_DATA_FILE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        help='mode you want to use, can be one of the following: '
                             'preprocess, generate_vectors, cluster')
    parser.add_argument('--vector_type',
                        default='TFIDF',
                        help='which type of vectors should be used, default: TFIDF.'
                             ' Possible values: RawTermFrequency, TFIDF, TFIDFTop100, TFIDFOnlyNouns.'
                             ' Only useful if mode is generate_vectors or cluster')
    parser.add_argument('--dataset_path',
                        default=news_dataset_file_path,
                        help='absolute path of the dataset you want to use, default: '
                             '{path to project}/data/raw/COVID Fake News Data.csv')
    parser_args = parser.parse_args()

    # Intermediate file names
    # Dataset file names
    news_dataset_preprocessed_file_name = os.path.join(dirname, conf.FAKE_NEWS_PREPROCESSED_DATA_FILE_PATH)

    # File name for idf_map
    term_idf_map_file_name = os.path.join(dirname, conf.TERM_IDF_VALUES_FILE_PATH)

    # File names for vectors
    raw_term_frequency_vectors_file_name = os.path.join(dirname, conf.RAW_TERM_FREQ_VECTORS_FILE_PATH)
    tf_idf_vectors_file_name = os.path.join(dirname, conf.TFIDF_VECTORS_FILE_PATH)
    tf_idf_top_100_vectors_file_name = os.path.join(dirname, conf.TFIDF_TOP100_VECTORS_FILE_PATH)
    tf_idf_only_nouns_vectors_file_name = os.path.join(dirname, conf.TFIDF_ONLY_NOUNS_VECTORS_FILE_PATH)

    if parser_args.mode == 'preprocess':
        df = prep.read_dataset(parser_args.dataset_path)
        prep.preprocess_and_save(df, news_dataset_preprocessed_file_name)

    elif parser_args.mode == 'generate_vectors':
        df = prep.read_dataset_with_index(news_dataset_preprocessed_file_name, 'docId')

        if parser_args.vector_type == 'RawTermFrequency':
            vg.generate_raw_term_frequency_vectors(df, raw_term_frequency_vectors_file_name)
        elif parser_args.vector_type == 'TFIDF':
            vg.generate_tf_idf_vectors(df, tf_idf_vectors_file_name)
        elif parser_args.vector_type == 'TFIDFTop100':
            vg.generate_top_100_vectors(df, tf_idf_top_100_vectors_file_name)
        elif parser_args.vector_type == 'TFIDFOnlyNouns':
            vg.generate_only_nouns_vectors(df, tf_idf_only_nouns_vectors_file_name)

    elif parser_args.mode == 'cluster':
        vector_dataframes = dict()

        if parser_args.vector_type == 'RawTermFrequency':
            vector_dataframes['RawTermFrequency'] = prep.read_dataset(raw_term_frequency_vectors_file_name)
        elif parser_args.vector_type == 'TFIDF':
            vector_dataframes['TFIDF'] = prep.read_dataset(tf_idf_vectors_file_name)
        elif parser_args.vector_type == 'TFIDFTop100':
            vector_dataframes['TFIDFTop100'] = prep.read_dataset(tf_idf_top_100_vectors_file_name)
        elif parser_args.vector_type == 'TFIDFOnlyNouns':
            vector_dataframes['TFIDFOnlyNouns'] = prep.read_dataset(tf_idf_only_nouns_vectors_file_name)

        for key in vector_dataframes.keys():
            df = vector_dataframes[key]
            df.drop(columns=['docId'], inplace=True)

            print('Running experiments on vectors of ', key)

            # Run k_means with 10, 50 and 100 clusters
            clustering.run_k_means(df, 10)
            clustering.run_k_means(df, 50)
            clustering.run_k_means(df, 100)

            # Run DBSCAN with different epsilon and minimum samples hyper-parameters
            clustering.run_dbscan(df, 1, 2)
            clustering.run_dbscan(df, 2, 2)
            clustering.run_dbscan(df, 2, 3)
