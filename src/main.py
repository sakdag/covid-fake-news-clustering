import os

import src.config.config as conf
import src.preprocessing.preprocessing as prep
import src.preprocessing.vector_generation as vg

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    news_dataset_file_name = os.path.join(dirname, conf.FAKE_NEWS_DATA_FILE_PATH)
    news_dataset_preprocessed_file_name = os.path.join(dirname, conf.FAKE_NEWS_PREPROCESSED_DATA_FILE_PATH)
    raw_term_frequency_vectors_file_name = os.path.join(dirname, conf.RAW_TERM_FREQ_VECTORS_FILE_PATH)
    term_idf_map_file_name = os.path.join(dirname, conf.TERM_IDF_VALUES_FILE_PATH)

    # df = prep.read_dataset(news_dataset_file_name)
    # prep.preprocess_and_save(df, news_dataset_preprocessed_file_name)

    df = prep.read_dataset(news_dataset_preprocessed_file_name)
    vg.generate_tf_idf_vectors(df, term_idf_map_file_name)
