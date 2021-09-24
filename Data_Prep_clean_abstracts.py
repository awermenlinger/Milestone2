import pandas as pd
import numpy as np
import re, string
import nltk   #nltk.download('stopwords')
from nltk.corpus import stopwords
import tqdm

def clean_extracts(extracts_file):
    clean_df = pd.read_csv(extracts_file)
    stop_words = stopwords.words('english')
    punct_remove = str.maketrans("", "", "!()[]{};:,<>./?@#$%^&*_~")
    clean_df.abstract = clean_df.abstract.astype('str') 
    for i, row in clean_df.iterrows():
        abstract = row.abstract
        if isinstance(abstract, str): 
            abstract = abstract.lower()
            abstract = abstract.translate(punct_remove)
            abstract = " ".join(word for word in abstract.split() if word not in (stop_words))
            clean_df.at[i,'abstract'] = abstract
        else: #debugging errors
            print(row["pubmed_id"])
            print(type(row.abstract))
            print(row.abstract)
            break    
    return clean_df

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("extracts_file", help="The path to the extract_articles_full.csv file")
    parser.add_argument("output_file", help="The path to the output CSV file")
    args = parser.parse_args()

    result = clean_extracts(args.extracts_file)
    result.to_csv(args.output_file, index=False)