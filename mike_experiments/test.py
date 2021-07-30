# with open("../pubmed_ids_extracted_abstracts.csv") as file:
#     lines = file.readlines()
#
# print(lines)

import pandas as pd

df = pd.read_csv("../pubmed_ids_extracted_abstracts.csv")
print(df)
