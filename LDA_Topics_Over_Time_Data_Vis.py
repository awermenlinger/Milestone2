import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw_df = pd.DataFrame()

##Save the LDA Model
filename = 'data/trained_lda_top3_output_df.sav'
raw_df = pickle.load(open(filename, 'rb'))

transformed_df = raw_df.copy()

# #if you use the top 3
# for i in range(1,21):
#     column_name = "t" + str(i)
#     transformed_df[column_name] = 0
#     transformed_df.loc[transformed_df['topic1'] == i , column_name] = 1
#     transformed_df.loc[transformed_df['topic2'] == i , column_name] = 1
#     transformed_df.loc[transformed_df['topic3'] == i , column_name] = 1

#if you use the top
for i in range(1,21):
    column_name = "t" + str(i)
    transformed_df[column_name] = 0
    transformed_df.loc[transformed_df['topic1'] == i , column_name] = 1


transformed_df.drop(columns=["pubmed_id", "topic1", "topic2", "topic3"], inplace=True)
#EDA: Very little data before 1975
transformed_df = transformed_df[transformed_df["created_date"] >= "1975-01-01"]
transformed_df.set_index(["created_date"], inplace = True)
transformed_df = transformed_df.resample("Y").sum()

#print(transformed_df.head(50))
sns.lineplot(data=transformed_df)
# x = np.linspace(0,46,46)
# plt.plot(transformed_df.index, transformed_df["t1"], "r", label="t1")
# plt.plot(transformed_df.index, transformed_df["t2"], "b", label="t2")
plt.show()

# raw_df.set_index(["created_date"], inplace = True)
# raw_df.to_csv("temp.csv")