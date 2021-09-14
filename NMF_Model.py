import pickle
import pandas as pd
from gensim.corpora import dictionary
#from gensim.models.nmf import Nmf #generating nothing....
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

num_topics=19

print("loading pre-processed texts...")
texts_file = 'models/bi_trained_lda_texts.sav'
abstracts = pickle.load(open(texts_file, 'rb'))

abstracts_sentences = [' '.join(text) for text in abstracts]

print("creating tfidf vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.01, max_features=5000, norm="l1")

abstracts_tfidf = tfidf_vectorizer.fit_transform(abstracts_sentences)

model = NMF(n_components=num_topics, init='nndsvd')

print("fitting model...")
model.fit(abstracts_tfidf)

#from: https://gist.github.com/ravishchawla/3f346318b85fa07196b761443b123bba
def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = tfidf_vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

print("getting results...")
results_NMF = get_nmf_topics(model, 15)

print("saving...")
result_file = 'models/bi_trained_NMF_model_sklearn_df.sav'
pickle.dump(results_NMF, open(result_file, 'wb'))

print("transforming original abstracts to get model weights...")
abstracts_tfidf_nmf = model.transform(abstracts_tfidf)

result_arrays = 'models/bi_trained_NMF_model_sklearn_res_array.npy'
np.save(result_arrays, abstracts_tfidf_nmf)

print(abstracts_tfidf_nmf[0])
print(type(abstracts_tfidf_nmf))
print(abstracts_tfidf_nmf.shape)