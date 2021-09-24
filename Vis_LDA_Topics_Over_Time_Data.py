import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

dates_df = input_data.copy()
dates_df = dates_df[["pubmed_id", "created_date"]] 
#dates_df.to_csv("dates.csv", index=False)


# print("get weights")
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
df.to_csv("temp.csv", index=False)

#df = pd.read_csv("temp.csv")

df.reset_index(inplace=True)
dates_df.reset_index(inplace=True)

transformed_df = dates_df.join(df)

#EDA: Very little data before 1975
transformed_df['created_date'] =  pd.to_datetime(dates_df['created_date'], format="%m/%d/%Y %H:%M")
transformed_df = transformed_df[transformed_df["created_date"] >= "1975-01-01"]
transformed_df = transformed_df.resample("Y").sum()
transformed_df.drop(columns=["pubmed_id"], inplace=True)
transformed_df.set_index(["created_date"], inplace = True)


print (transformed_df.head())
# transformed_df.columns[[topics]] = transformed_df.columns[[topics]].apply(lambda x: x/x.sum(), axis=1)

#print(transformed_df.head(50))
#sns.lineplot(data=df)
# x = np.linspace(0,46,46)
# plt.plot(transformed_df.index, transformed_df["t1"], "r", label="t1")
# plt.plot(transformed_df.index, transformed_df["t2"], "b", label="t2")
#plt.show()

# raw_df.set_index(["created_date"], inplace = True)
# raw_df.to_csv("temp.csv")