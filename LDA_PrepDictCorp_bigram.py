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

bi_dic_file = "models/bi_trained_lda_dictionary.sav"
bi_corp_file = "models/bi_trained_lda_corpus.sav"
bi_model_file = "models/bi_trained_lda.sav"
bi_text_file = "models/bi_trained_lda_texts.sav"
bi_tfidf_corp_file = "models/bi_trained_lda_corpus_tfidf.sav"
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

print ("Preprocessing the abstracts")
doc_processed = input_data['abstract'].map(preprocess)

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
doc_processed_bigram = list(doc_processed)

bigram = Phrases(doc_processed_bigram, min_count=20)

for idx in range(len(doc_processed_bigram)):
    for token in bigram[doc_processed_bigram[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            doc_processed_bigram[idx].append(token)

pickle.dump(doc_processed_bigram, open(bi_text_file, 'wb'))

print ("Building the dictionary")
dictionary = corpora.Dictionary(doc_processed)
#save the dictionary
pickle.dump(dictionary, open(bi_dic_file, 'wb'))      

print ("Building the corpus")
corpus = [dictionary.doc2bow(doc) for doc in doc_processed]
#save the corpus
pickle.dump(corpus, open(bi_corp_file, 'wb')) 

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#save the tfidf_corpus
pickle.dump(corpus_tfidf, open(bi_tfidf_corp_file, 'wb'))