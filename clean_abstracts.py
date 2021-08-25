import pandas as pd
import numpy as np
import re, string
import nltk   #nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_extracts(extracts_file):
    clean_df = pd.read_csv(extracts_file)
    
    #lower case everything
    clean_df.Abstract = clean_df.Abstract.str.lower()

    #remove punctuation
    clean_df.Abstract = clean_df.Abstract.str.replace('[{}]'.format(string.punctuation), '')

    #remove stopwords
    stop_words = stopwords.words('english')
    clean_df.Abstract = clean_df.Abstract.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Remove single quotes
    #clean_df.Abstract = clean_df.Abstract.apply(lambda x: re.sub("\'", "", x))
    #not sure if necessary
    
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