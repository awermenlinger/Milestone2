import pandas as pd
import numpy as np
import re, string
import nltk   #nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.utils import tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(cleaned_extracts_file):
    tokenized_df = pd.read_csv(cleaned_extracts_file)
    print("tokenizing...")
    tokenized_df.abstract = tokenized_df.abstract.apply(tokenize)
    return tokenzied_df


def create_tfidf_vectorizer(tokenized_df):
    print("vectorizing...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3)).fit(tokenzied_df.abstract)
    
    return tfidf_vectorizer

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned_extracts_file", help="The path to the pubmed_ids_cleaned_abstracts.csv file")
    parser.add_argument("output_file", help="The path to the output CSV file")
    args = parser.parse_args()

    result = create_tfidf_vectorizer(tokenize(args.cleaned_extracts_file))
    print("saving...")
    with open(args.output_file, 'wb') as fout:
        pickle.dump(result, fout)