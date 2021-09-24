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
#output_data['created_date'] = output_data['created_date'].str[:-6]
output_data['created_date'] =  pd.to_datetime(output_data['created_date'], format="%m/%d/%Y %H:%M")

print("get weights")
# https://stackoverflow.com/questions/62174945/gensim-extract-100-most-representative-documents-for-each-topic
topic_probs = ldamodel.get_document_topics(corpus) #get the list of topic probabilities by doc
topic_dict = [dict(x) for x in topic_probs] #convert to dictionary to convert to data frame
df = pd.DataFrame(topic_dict).fillna(0) #convert to data frame, fill topics < 0.01 as 0

# this was needed when loading from csv ... not required when running in line
# columns_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
#                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
# df = df.reindex(columns_names, axis=1)
df = df.reindex(sorted(df.columns), axis=1)

maxValueIndex = df.idxmax()

print("find max doc and summarize")
summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-pubmed")
summary_per_topic = []
raw_text_file = "models/raw_texts.sav"
abstracts = pickle.load(open(raw_text_file, 'rb'))

i=0
for top_abs_topic in tqdm(maxValueIndex):
    
    abstract = abstracts[top_abs_topic]
    summary_text = summarizer(abstract, max_length=100, min_length=5, do_sample=False)[0]['summary_text']    
    topic = str(i) + ": " + summary_text
    summary_per_topic.append(topic)
    i+=1

with open('results/LDA_topics.txt', 'w') as f:
    for item in summary_per_topic:
        f.write("%s\n\n" % item)


# @misc{zaheer2021big,
#       title={Big Bird: Transformers for Longer Sequences}, 
#       author={Manzil Zaheer and Guru Guruganesh and Avinava Dubey and Joshua Ainslie and Chris Alberti and Santiago Ontanon and Philip Pham and Anirudh Ravula and Qifan Wang and Li Yang and Amr Ahmed},
#       year={2021},
#       eprint={2007.14062},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }


x=ldamodel.show_topics(num_topics=21, num_words=10,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

dict_keywords = {}

#Below Code Prints Topics and Words
for topic,words in topics_words:
    dict_keywords[topic] = words

df = pd.DataFrame.from_dict(dict_keywords)
df.to_csv("results/LDA_topics.csv")