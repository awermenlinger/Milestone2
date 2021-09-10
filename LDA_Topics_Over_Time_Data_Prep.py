from gensim.corpora import dictionary
import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os
import logging
from tqdm import tqdm

# load the model from disk
filename = 'models/trained_lda.sav'
ldamodel = pickle.load(open(filename, 'rb'))

# filename = 'models/trained_lda_dictionary.sav'
# dictionary = pickle.load(open(filename, 'rb'))

filename = 'models/trained_lda_corpus.sav'
corpus = pickle.load(open(filename, 'rb'))

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
input_data = input_data[["pubmed_id", "created_date"]] 
output_data = input_data.copy()
#output_data['created_date'] = output_data['created_date'].str[:-6]
output_data['created_date'] =  pd.to_datetime(output_data['created_date'], format="%m/%d/%Y %H:%M")
output_data["topic1"] = 0
output_data["topic2"] = 0
output_data["topic3"] = 0

for i in tqdm(range(0,len(output_data))):
    topics = ldamodel[corpus[i]]
    top3_topics = sorted(topics, reverse=True, key=lambda t: t[1])[:3]
    top3_topics= [t[0] for t in top3_topics]
    try:
        output_data.loc[i, 'topic1'] = top3_topics[0]
    except IndexError:
        output_data.loc[i, 'topic1'] = 0
    
    try:
        output_data.loc[i, 'topic2'] = top3_topics[1]
    except IndexError:
        output_data.loc[i, 'topic2'] = 0

    try:
        output_data.loc[i, 'topic3'] = top3_topics[2]
    except IndexError:
        output_data.loc[i, 'topic3'] = 0

#Save the data
filename = 'data/trained_lda_top3_output_df.sav'
pickle.dump(output_data, open(filename, 'wb'))       