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

input_data.abstract = input_data.abstract.astype('str')                

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


print("create new df")

# Topics per decade
Tot = 6
Cols = 3

# Compute Rows required
Rows = Tot // Cols 
Rows += Tot % Cols

# Create a Position index
Position = range(1,Tot + 1)

for column in df.columns:
  df[column] = np.where(df[column].values <= 0.05, 0, 1)

transformed_df = dates_df.join(df)
transformed_df.reset_index(drop=True, inplace=True)

#EDA: Very little data before 1975
transformed_df['created_date'] =  pd.to_datetime(transformed_df['created_date'], format="%m/%d/%Y %H:%M")
transformed_df = transformed_df[transformed_df["created_date"] >= "1970-01-01"]
transformed_df.drop(columns=["pubmed_id"], inplace=True)
transformed_df.set_index(["created_date"], inplace = True)
transformed_df = transformed_df.resample("10AS").sum()
transformed_df = transformed_df.div(transformed_df.sum(axis=1), axis=0)
transformed_df["decades"] = ["70s", "80s", "90s", "00s", "10s", "20s"]
transformed_df.set_index("decades", inplace=True)
transformed_df = transformed_df.T
transformed_df.head(10)


column_names = transformed_df.columns
# Create main figure

fig = plt.figure(1,figsize=(15,8))

for k in range(Tot):

  # add every single subplot to the figure with a for loop
  x = transformed_df.index
  y = transformed_df.iloc[:, k]
  title = column_names[k]
  ax = fig.add_subplot(Rows,Cols,Position[k]);
  ax.set_title(title);
  ax.tick_params(labelrotation=90);
  ax.axhline(y=0.05,linewidth=1, color='r')
  ax.bar(x,y);      # Or whatever you want in the subplot
fig.tight_layout();
plt.savefig('results/TopicsOverTime_DecadesDistribution.png')
plt.show()
