import pandas as pd
import gensim
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS

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

# SETTINGS FOR MODEL
RANDOM_SEED = 7245
passes = 5
num_topics=10

addtl_stop_words = ["patient", "patients", "group", "groups" "placebo", "survival", "treatment", "response", "remission",
                     "day", "days", "week", "weeks", "month", "months", "year", "years", "median"]
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

def train_model(data_files, dic_file, corp_file):
   input_data = pd.DataFrame()
   print ("loading the files")
   for file in data_files:
      df_load = pd.read_csv(file,skip_blank_lines=True)
      input_data = input_data.append(df_load)
      # df1 = pd.read_csv("data/pubmed_articles_cancer_01_smaller.csv",skip_blank_lines=True)
      # df1.dropna(inplace=True, axis = 0, how = 'all')
      # df2 = pd.read_csv("data/pubmed_articles_cancer_02_smaller.csv")
      # df2.dropna(inplace=True, axis = 0, how = 'all')
      # df3 = pd.read_csv("data/pubmed_articles_cancer_03_smaller.csv")
      # df3.dropna(inplace=True, axis = 0, how = 'all')
      # df4 = pd.read_csv("data/pubmed_articles_cancer_04_smaller.csv")
      # df4.dropna(inplace=True, axis = 0, how = 'all')
      # input_data = pd.DataFrame().append([df1,df2,df3, df4])

   input_data.abstract = input_data.abstract.astype('str')

   print ("Preprocessing the abstracts")
   doc_processed = input_data['abstract'].map(preprocess)

   print ("Building the dictionary")
   dictionary = corpora.Dictionary(doc_processed)
   #save the dictionary
   pickle.dump(dictionary, open(dic_file, 'wb'))      

   print ("Building the corpus")
   corpus = [dictionary.doc2bow(doc) for doc in doc_processed]
   #save the corpus
   pickle.dump(corpus, open(corp_file, 'wb'))      

   print ("Training the model")
   LDA = gensim.models.ldamodel.LdaModel

   #Lda model with settings
   ldamodel = LDA(corpus, num_topics=num_topics, id2word = dictionary, passes=passes, random_state=RANDOM_SEED)
   return ldamodel
                                     

# COULDN'T GET MAKEFILE TO WORK IN VSCODE... check later
# if __name__ == "__main__":
#     import argparse
#     import pandas as pd
#     import pickle

#     parser = argparse.ArgumentParser()
#     parser.add_argument("dic_file", default="models/trained_lda_dictionary.sav")
#     parser.add_argument("corp_file", default="models/trained_lda_corpus.sav")
#     parser.add_argument("model_file", default="models/trained_lda.sav")
#     parser.add_argument("--data_files", help="list of data files for model", default=["data/pubmed_articles_cancer_01_smaller.csv",
#                         "data/pubmed_articles_cancer_02_smaller.csv", "data/pubmed_articles_cancer_03_smaller.csv",
#                         "data/pubmed_articles_cancer_04_smaller.csv"])
    
#     args = parser.parse_args()

#     ldamodel = train_model(args.data_files, args.dic_file, args.corp_file)
#     #Save the LDA Model
#     pickle.dump(ldamodel, open(args.model_file, 'wb'))

dic_file = "models/trained_lda_dictionary.sav"
corp_file = "models/trained_lda_corpus.sav"
model_file = "models/trained_lda.sav"
data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]


ldamodel = train_model(data_files, dic_file, corp_file)
#Save the LDA Model
pickle.dump(ldamodel, open(model_file, 'wb'))   