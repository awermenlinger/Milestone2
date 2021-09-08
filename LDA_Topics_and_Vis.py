import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os
import logging

#http://www.cse.chalmers.se/~richajo/dit862/L13/LDA%20with%20gensim%20(small%20example).html
# for gensim to output some progress information while it's training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  


#Some code inspired from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

# SETTINGS FOR MODEL
RANDOM_SEED = 7245
passes = 10
num_topics=20

def preprocess(text):
   result = []
   for token in gensim.utils.simple_preprocess(text):
      if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
         result.append(token)
   return result


#load the files
df1 = pd.read_csv("data/pubmed_articles_cancer_01_smaller.csv",skip_blank_lines=True)
df1.dropna(inplace=True, axis = 0, how = 'all')
df2 = pd.read_csv("data/pubmed_articles_cancer_02_smaller.csv")
df2.dropna(inplace=True, axis = 0, how = 'all')
df3 = pd.read_csv("data/pubmed_articles_cancer_03_smaller.csv")
df3.dropna(inplace=True, axis = 0, how = 'all')
df4 = pd.read_csv("data/pubmed_articles_cancer_04_smaller.csv")
df4.dropna(inplace=True, axis = 0, how = 'all')
input_data = pd.DataFrame().append([df1,df2,df3, df4])

#ensure abstracts are strings
input_data.abstract = input_data.abstract.astype('str')

#preprocess abstracts
doc_processed = input_data['abstract'].map(preprocess)

#build the dictionary
dictionary = corpora.Dictionary(doc_processed)
#save the dictionary
filename = 'models/trained_lda_dictionary.sav'
pickle.dump(dictionary, open(filename, 'wb'))      

#to prepapre a document term matrix
corpus = [dictionary.doc2bow(doc) for doc in doc_processed]
#save the corpus
filename = 'models/trained_lda_corpus.sav'
pickle.dump(corpus, open(filename, 'wb'))      

#Lda model
LDA = gensim.models.ldamodel.LdaModel

#Lda model with settings
ldamodel = LDA(corpus, num_topics=num_topics, id2word = dictionary, passes=passes, random_state=RANDOM_SEED)

#Save the LDA Model
filename = 'models/trained_lda.sav'
pickle.dump(ldamodel, open(filename, 'wb'))                                        

#Visualize the topics
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

LDAvis_prepared = pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary)
with open(LDAvis_data_filepath, 'wb') as f:
   pickle.dump(LDAvis_prepared, f)

#Load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
   LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html') 