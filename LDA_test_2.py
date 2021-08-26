import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os

#Some code inspired from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0



def preprocess(text):
   result = []
   for token in gensim.utils.simple_preprocess(text):
      if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
         result.append(token)
   return result

num_topics=10
original_df = pd.read_csv("data/pubmed_articles.csv")
original_df.abstract = original_df.abstract.astype('str')
input_data = original_df.copy()

doc_processed = input_data['abstract'].map(preprocess)

dictionary = corpora.Dictionary(doc_processed)
#to prepapre a document term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_processed]

#Lda model
Lda = gensim.models.ldamodel.LdaModel
#Lda model to get the num_topics, number of topic required, 
#passses is the number training do you want to perform
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=2)

# Visualize the topics
#pyLDAvis.enable_notebook() only works in ipython - will save to html
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

LDAvis_prepared = pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html') 