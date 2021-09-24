import gensim
from gensim.parsing.preprocessing import STOPWORDS
import pickle
import logging
#import nltk
#nltk.download('wordnet')

#https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py

# SETTINGS FOR MODEL
RANDOM_SEED = 7245
dic_file = "models/trained_lda_dictionary.sav"
corp_file = "models/trained_lda_corpus.sav"
model_file = "models/trained_hdp.sav"
#for gensim to output some progress information while it's training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print ("Loading the dic, corpus and model")
dictionary = pickle.load(open(dic_file, 'rb'))
corpus = pickle.load(open(corp_file, 'rb'))  
print ("Training the model")

hdpmodel = gensim.models.HdpModel(corpus, id2word=dictionary) 

#Save the HDP Model
pickle.dump(hdpmodel, open(model_file, 'wb'))