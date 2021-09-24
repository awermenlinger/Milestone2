import pickle
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
from tqdm import tqdm
from gensim.models.ldamulticore import LdaMulticore
from pprint import pprint

# load the model from disk
filename = 'models/trained_lda.sav'
ldamodel = LdaMulticore.load(filename)

filename = 'models/trained_lda_corpus.sav'
corpus = pickle.load(open(filename, 'rb'))

#load the files
data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print("load the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')                

output_data = input_data.copy()
output_data = output_data[["pubmed_id", "created_date"]] 
input_data = input_data.abstract.astype('str')
output_data['created_date'] =  pd.to_datetime(output_data['created_date'], format="%m/%d/%Y %H:%M")

print("get weights")
# https://stackoverflow.com/questions/62174945/gensim-extract-100-most-representative-documents-for-each-topic
topic_probs = ldamodel.get_document_topics(corpus) #get the list of topic probabilities by doc
topic_dict = [dict(x) for x in topic_probs] #convert to dictionary to convert to data frame
df = pd.DataFrame(topic_dict).fillna(0) #convert to data frame, fill topics < 0.01 as 0
df = df.reindex(sorted(df.columns), axis=1)

columns_names = ["infection risk", "thyroid cancer", "safety and efficacy", "leukemia chemotherapy", "surgical intervention",
                "lymph nodes detection", "pain management", "cervical cancer", "bladder cancer",  "risk prediction",
                "adjuvant therapy", "healthy habits", "hematologic toxicity", "surgical complications", "tumor angiogenesis",
                "Intraoperative Radiation Therapy", "radiotherapy", "stem cell transplantation", "glioma", "behavioral intervention",
                "prostate cancer"]
df.columns = columns_names

pprint (df.head())