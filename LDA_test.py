import pandas as pd
import numpy as np
import gensim
import re, string
import nltk   #nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.corpora as corpora
from pprint import pprint


def generate_topics(cleaned_extracts_file):
    clean_df = pd.read_csv(cleaned_extracts_file)
    clean_df.abstract = clean_df.abstract.astype('str') 
    id2word = corpora.Dictionary(clean_df.abstract)
    texts = clean_df.abstract
    corpus = [id2word.doc2bow(text) for text in texts]
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]



    return doc_lda

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned_extracts_file", help="The path to the pubmed_ids_cleaned_abstracts.csv file")
    parser.add_argument("output_file", help="The path to the output pkl model file")
    args = parser.parse_args()

    result = generate_topics(args.cleaned_extracts_file)
    print("saving...")
    with open(args.output_file, 'wb') as fout:
        pickle.dump(result, fout)