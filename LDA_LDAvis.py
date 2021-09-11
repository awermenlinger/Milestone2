import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os

#SETTINGS
num_topics=10

print ("Loading the dic, corpus and model")
filename = 'models/trained_lda_dictionary.sav'
dictionary = pickle.load(open(filename, 'rb'))

filename = 'models/trained_lda_corpus.sav'
corpus = pickle.load(open(filename, 'rb'))

filename = 'models/trained_lda.sav'
ldamodel = pickle.load(open(filename, 'rb'))

#Visualize the topics
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

LDAvis_prepared = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
with open(LDAvis_data_filepath, 'wb') as f:
   pickle.dump(LDAvis_prepared, f)

#Load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
   LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html') 