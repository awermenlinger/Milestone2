# goal is to extract the abstracts from pub_med_ids and create a dataframe with
# PubmedID, Date Published, Abstract
from Bio import Entrez, Medline
import pandas as pd
import numpy as np

Entrez.email = "awerm@umich.edu"

API = input ("Enter api key :")
Entrez.api_key = API




df_extracted = pd.read_csv("pubmed_ids_from_search - small.txt", header = None)
df_extracted.columns = ["pubmed_id"]
df_extracted["Title"] = "Sample title"
df_extracted["CreatedDate"] = "1990-01-01"
df_extracted["Abstract"] = "Sample abstract"


for i, row in df_extracted.iterrows():
    pubmed_id = row["pubmed_id"]
    handle = Entrez.efetch(db="pubmed",  rettype="medline", retmode="text", id=pubmed_id)
    article = Medline.parse(handle)
    article = list(article)
    title = article[0].get("TI", np.nan)
    if title != np.nan : title = title.replace('[', '').replace('].', '')
    abstract = article[0].get("AB", np.nan)
    createddate = article[0].get("CRDT", np.nan)
    df_extracted.at[i,'Title'] = title
    df_extracted.at[i,'Abstract'] = abstract
    df_extracted.at[i,'CreatedDate'] = createddate


df_extracted.to_csv("pubmed_ids_extracted_abstracts.csv", index = False)