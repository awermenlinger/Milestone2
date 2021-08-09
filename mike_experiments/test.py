# with open("../pubmed_ids_extracted_abstracts.csv") as file:
#     lines = file.readlines()
#
# print(lines)

# import pandas as pd
#
# df = pd.read_csv("../pubmed_ids_extracted_abstracts.csv")
# print(df)

from Bio import Entrez, Medline
from mike_experiments.config import *

Entrez.email = mike_email
Entrez.api_key = mike_api_key




pbmdid = 34281299

handle = Entrez.efetch(db='pubmed', rettype='medline', retmode='text', id=pbmdid)
article = Medline.parse(handle)
article = list(article)
key_words = article[0].get('OT')
mesh_terms = article[0].get('MH')
print(article[0].get('MH'))
