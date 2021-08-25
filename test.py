import pandas as pd
# import sklearn as sk
# input = "pubmed_ids_cleaned_abstracts.csv"
# tokenized_df = pd.read_csv(input)
# print(tokenized_df.head())

test_df = pd.read_csv("data/pubmed_articles.csv")
print(test_df[test_df["pubmed_id"]==34281299])
print(test_df[test_df["pubmed_id"]==34281299].info)