import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from gensim.models.ldamulticore import LdaMulticore
import numpy as np

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

dates_df = input_data.copy()
dates_df = dates_df[["pubmed_id", "created_date"]] 

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

#needed for the join to work properly because dates_df Index is Int64 and sequential, while the other is a range
dates_df.reset_index(drop=True, inplace=True)

print("create new df")
transformed_df = dates_df.join(df)
transformed_df.reset_index(drop=True, inplace=True)

#EDA: Very little data before 1975
transformed_df['created_date'] =  pd.to_datetime(transformed_df['created_date'], format="%m/%d/%Y %H:%M")
transformed_df = transformed_df[transformed_df["created_date"] >= "1975-01-01"]
transformed_df.drop(columns=["pubmed_id"], inplace=True)
transformed_df.set_index(["created_date"], inplace = True)
transformed_df = transformed_df.resample("Q").sum()

transformed_df = transformed_df.div(transformed_df.sum(axis=1), axis=0)

print("plot and save")
# Some code used from: https://towardsdatascience.com/dynamic-subplot-layout-in-seaborn-e777500c7386
# Subplots are organized in a Rows x Cols Grid
# Tot and Cols are known

Tot = 21
Cols = 7

# Compute Rows required

Rows = Tot // Cols 
Rows += Tot % Cols

# Create a Position index

Position = range(1,Tot + 1)


column_names = transformed_df.columns

# Create main figure
# Same y axes
fig = plt.figure(1, figsize=(21, 7.0))

for k in range(Tot):

  # add every single subplot to the figure with a for loop
  x = transformed_df.index
  y = transformed_df.iloc[:, k]
  title = column_names[k]
  ax = fig.add_subplot(Rows,Cols,Position[k]);
  ax.set_title(title);
  ax.set_ylim(0, 0.3);
  date_form = DateFormatter("'%y");
  ax.xaxis.set_major_formatter(date_form);
  ax.plot(x,y);      # Or whatever you want in the subplot
fig.tight_layout();
plt.savefig('results/TopicsOverTime_SameYAxis.png')


# Different y axes
fig = plt.figure(2, figsize=(21.0, 7.0))

for k in range(Tot):

  # add every single subplot to the figure with a for loop
  x = transformed_df.index
  y = transformed_df.iloc[:, k]
  title = column_names[k]
  ax = fig.add_subplot(Rows,Cols,Position[k]);
  ax.set_title(title);
  date_form = DateFormatter("'%y");
  ax.xaxis.set_major_formatter(date_form);
  ax.plot(x,y);      # Or whatever you want in the subplot
fig.tight_layout();
plt.savefig('results/TopicsOverTime_OwnYAxis.png')