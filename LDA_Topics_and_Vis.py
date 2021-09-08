import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os


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
df1 = pd.read_csv("data/pubmed_articles_cancer_01.csv",skip_blank_lines=True)
df1.dropna(inplace=True, axis = 0, how = 'all')
df2 = pd.read_csv("data/pubmed_articles_cancer_02.csv")
df2.dropna(inplace=True, axis = 0, how = 'all')
df3 = pd.read_csv("data/pubmed_articles_cancer_03.csv")
df3.dropna(inplace=True, axis = 0, how = 'all')
df4 = pd.read_csv("data/pubmed_articles_cancer_04.csv")
df4.dropna(inplace=True, axis = 0, how = 'all')
input_data = pd.DataFrame().append([df1,df2,df3, df4])

#ensure abstracts are strings
input_data.abstract = input_data.abstract.astype('str')

#preprocess abstracts
doc_processed = input_data['abstract'].map(preprocess)

#build the dictionary
dictionary = corpora.Dictionary(doc_processed)

#to prepapre a document term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_processed]

#Lda model
LDA = gensim.models.ldamodel.LdaModel

#Lda model with settings
ldamodel = LDA(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=passes, random_state=RANDOM_SEED)

#Save the LDA Model
filename = 'models/trained_lda.sav'
pickle.dump(ldamodel, open(filename, 'wb'))


# Visualize the topics
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

LDAvis_prepared = pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html') 