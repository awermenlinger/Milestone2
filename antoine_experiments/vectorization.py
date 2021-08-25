import pandas as pd
import re
import string
from nltk.corpus import stopwords

clean_df = pd.read_csv("Milestone2/pubmed_ids_extracted_abstracts.csv")
clean_df.Abstract = clean_df.Abstract.str.lower()
clean_df.Abstract = clean_df.Abstract.str.replace('[{}]'.format(string.punctuation), '')
stop_words = stopwords.words('english')
clean_df.Abstract = clean_df.Abstract.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
clean_df.Abstract = clean_df.Abstract.apply(lambda x: re.sub("\'", "", x))
print(clean_df.head())