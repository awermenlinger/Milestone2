import pickle
from gensim.corpora import dictionary
from gensim.models import NMF_Model
from gensim.utils import chunkize

print("Loading pre-processed texts / tfidf")
tf_idf_file = 'models/bi_trained_lda_corpus_tfidf.sav'
tfidf_corpus = pickle.load(open(tf_idf_file, 'rb'))

dictionary = 'models/bi_trained_lda_dictionary.sav'
tfidf_corpus = pickle.load(open(tf_idf_file, 'rb'))


nmf = NMF_Model(corpus=tfidf_corpus, num_topics=19, id2word=dictionary,
                 chunkize=2000, passes=10, random_state=7245)

result_file = 'models/bi_trained_NMF_model.sav'
pickle.dump(result_file, open(result_file, 'wb'))