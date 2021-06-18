import os
import sys

import src.config.config as conf
import src.preprocessing.preprocessing as prep
import src.preprocessing.vector_generation as vg
import src.clustering.clustering as clustering

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    mode = str(sys.argv[1])

    # Possible values for vectors to work on: RawTermFrequency TFIDF TFIDFTop100 TFIDFOnlyNouns
    vectors_to_work_on = list()

    for i in range(2, len(sys.argv)):
        vectors_to_work_on.append(sys.argv[i])

    # Dataset file names
    news_dataset_file_name = os.path.join(dirname, conf.FAKE_NEWS_DATA_FILE_PATH)
    news_dataset_preprocessed_file_name = os.path.join(dirname, conf.FAKE_NEWS_PREPROCESSED_DATA_FILE_PATH)

    # File name for idf_map
    term_idf_map_file_name = os.path.join(dirname, conf.TERM_IDF_VALUES_FILE_PATH)

    # File names for vectors
    raw_term_frequency_vectors_file_name = os.path.join(dirname, conf.RAW_TERM_FREQ_VECTORS_FILE_PATH)
    tf_idf_vectors_file_name = os.path.join(dirname, conf.TFIDF_VECTORS_FILE_PATH)
    tf_idf_top_100_vectors_file_name = os.path.join(dirname, conf.TFIDF_TOP100_VECTORS_FILE_PATH)
    tf_idf_only_nouns_vectors_file_name = os.path.join(dirname, conf.TFIDF_ONLY_NOUNS_VECTORS_FILE_PATH)

    if mode == 'preprocess':
        df = prep.read_dataset(news_dataset_file_name)
        prep.preprocess_and_save(df, news_dataset_preprocessed_file_name)

    elif mode == 'generate_vectors':
        df = prep.read_dataset_with_index(news_dataset_preprocessed_file_name, 'docId')

        for vectors_name in vectors_to_work_on:
            if vectors_name == 'RawTermFrequency':
                vg.generate_raw_term_frequency_vectors(df, raw_term_frequency_vectors_file_name)
            if vectors_name == 'TFIDF':
                vg.generate_tf_idf_vectors(df, tf_idf_vectors_file_name)
            if vectors_name == 'TFIDFTop100':
                vg.generate_top_100_vectors(df, tf_idf_top_100_vectors_file_name)
            if vectors_name == 'TFIDFOnlyNouns':
                vg.generate_only_nouns_vectors(df, tf_idf_only_nouns_vectors_file_name)

    elif mode == 'cluster':
        vector_dataframes = dict()

        for vectors_name in vectors_to_work_on:
            if vectors_name == 'RawTermFrequency':
                vector_dataframes['RawTermFrequency'] = prep.read_dataset(raw_term_frequency_vectors_file_name)
            if vectors_name == 'TFIDF':
                vector_dataframes['TFIDF'] = prep.read_dataset(tf_idf_vectors_file_name)
            if vectors_name == 'TFIDFTop100':
                vector_dataframes['TFIDFTop100'] = prep.read_dataset(tf_idf_top_100_vectors_file_name)
            if vectors_name == 'TFIDFOnlyNouns':
                vector_dataframes['TFIDFOnlyNouns'] = prep.read_dataset(tf_idf_only_nouns_vectors_file_name)

        for key in vector_dataframes.keys():
            df = vector_dataframes[key]
            df.drop(columns=['docId'], inplace=True)
            df.sample(frac=1)

            print('Running experiments on vectors of ', key)

            # Run k_means with 10, 50 and 100 clusters
            clustering.run_k_means(df, 10)
            # clustering.run_k_means(df, 50)
            # clustering.run_k_means(df, 100)

            # Run DBSCAN with different epsilon and minimum samples hyper-parameters
            clustering.run_dbscan(df, 3, 2)

            print("--------------------------------\n\n")
