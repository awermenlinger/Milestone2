import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from gensim.models.ldamulticore import LdaMulticore


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

dates_df = pd.read_csv("dates.csv")

dates_df['created_date'] =  pd.to_datetime(dates_df['created_date'], format="%m/%d/%Y %H:%M")
dates_df = dates_df[dates_df["created_date"] >= "1970-01-01"] #EDA: Very little data before 1975
dates_df = dates_df[dates_df["created_date"] < "2020-01-01"] #EDA: Second half of 2020 missing
dates_df.set_index(["created_date"], inplace = True)
dates_df.columns = ["articles"]
dates_df = dates_df.resample("Y").count()
dates_df.head(10)
sns.set(rc={'figure.figsize':(10,6)})
sns.lineplot(x=dates_df.index, y=dates_df.articles)
plt.show()