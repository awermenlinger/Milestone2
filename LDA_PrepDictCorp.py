import pandas as pd
import gensim
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pickle
import os
import logging
#import nltk
#nltk.download('wordnet')

#http://www.cse.chalmers.se/~richajo/dit862/L13/LDA%20with%20gensim%20(small%20example).html

# for gensim to output some progress information while it's training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  

#Some code inspired from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 & 
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

# SETTINGS FOR MODEL ---------------------------------------------------
RANDOM_SEED = 7245
chunk_size = 5000
passes = 5
num_topics=10

dic_file = "models/trained_lda_dictionary.sav"
corp_file = "models/trained_lda_corpus.sav"
model_file = "models/trained_lda.sav"
text_file = "models/trained_lda_texts.sav"
raw_text_file = "models/raw_texts.sav"
tfidf_corp_file = "models/trained_lda_corpus_tfidf.sav"
data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

#-----------------------------------------------------------------------

addtl_stop_words = ["patient", "patients", "group", "groups" "placebo", "survival", "treatment", "response", "remission",
                     "day", "days", "week", "weeks", "month", "months", "year", "years", "median", "dose", "doses", "result", "results",
                     "conclusion", "conclusions", "study", "significance", "significant", "arm", "arms", "random", "clinical",
                     "trial", "trials", "effect"]
stop_words = STOPWORDS.union(set(addtl_stop_words))

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
   return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
   result = []
   for token in gensim.utils.simple_preprocess(text):
      if token not in stop_words and len(token) > 2:
         result.append(lemmatize_stemming(token))
   
   return result

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')

docs = list(input_data['abstract'])
pickle.dump(docs, open(raw_text_file, 'wb'))

print ("Preprocessing the abstracts")
doc_processed = input_data['abstract'].map(preprocess)
pickle.dump(doc_processed, open(text_file, 'wb'))    

print ("Building the dictionary")
dictionary = corpora.Dictionary(doc_processed)
dictionary.filter_extremes(no_below=5, no_above=0.5)

#save the dictionary
pickle.dump(dictionary, open(dic_file, 'wb'))      

print ("Building the corpus")
corpus = [dictionary.doc2bow(doc) for doc in doc_processed]
#save the corpus
pickle.dump(corpus, open(corp_file, 'wb')) 

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#save the tfidf_corpus
pickle.dump(corpus_tfidf, open(tfidf_corp_file, 'wb'))
pickle.dump(corpus, open(corp_file, 'wb'))