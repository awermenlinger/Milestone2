import pandas as pd
from top2vec import Top2Vec
import pickle
from random import sample

data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')
doc_processed = list(input_data['abstract'])
doc_processed = sample(doc_processed, int(0.1*len(doc_processed)))

top2vec_file = "models/Top2Vec_model.sav"

model = Top2Vec(documents=doc_processed, embedding_model = "universal-sentence-encoder", speed="learn", workers=8)
#Save the Top2Vec Model
model.save("top2vec_file")

topic_words, word_scores, topic_nums = model.get_topics(19)

for topic in topic_nums:
    model.generate_topic_wordcloud(topic)