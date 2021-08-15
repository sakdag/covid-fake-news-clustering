# Covid Fake News Clustering
This project aims to utilize clustering methods to find fake Covid news. Natural Language Processing (NLP) steps are
applied to preprocess news, then using Information Retrieval methodologies, vector representation of the news are
created. In the end, clustering methods KMeans and DBSCAN are used to find fake Covid news.

## Usage
Following commands should be run in order to output clustering scores:

1. Run preprocessing steps.
```
python3 main.py preprocess
```

2. Generate vectors for news texts, that reside in preprocessed dataset outputted from 1st step, which can be found 
under /data/interim/FakeNewsDataPreprocessed.csv
```
python3 main.py generate_vectors --vector_type <type_of_vector_to_generate>
ex: python3 main.py generate_vectors --vector_type TFIDFTop100
```
There are 4 different type of vectors that can be selected.

- RawTermFrequency: Raw Term Frequency Vectors
- TFIDF: TF-IDF Vectors
- TFIDFTop100: TF-IDF Vectors where only Top 100 vectors in terms of TF-IDF scores is used.
- TFIDFOnlyNouns: TF-IDF Vectors where only nouns are used.

3. Run clustering algorithms on generated vectors outputted from 2nd step, which can be found under
/data/processed/<type_of_vectors_generated>Vectors.csv. Silhouette Coefficient scores for K-Means and DBSCAN 
is reported. For KMeans, results where k is 10, 50 and 100 is reported. For DBSCAN results where eps: 1 
and min_samples: 2, eps: 2 and min_samples: 2, eps: 2 and min_samples: 3 are reported. 
```
python3 main.py cluster --vector_type <type_of_vectors_to_run_clusters_on>
ex: python3 main.py cluster --vector_type RawTermFrequency
```

Note that you can always use -h option to learn more about parameters.

## Dataset

Dataset used in this project is [Covid Fake News Dataset](https://zenodo.org/record/4282522#.YRklitOA63J).

## Environment

Pyhton 3.9 and Conda environment with dependencies as given in requirements.txt is used.

## License
[MIT](https://choosealicense.com/licenses/mit/)
