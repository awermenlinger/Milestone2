from bertopic import BERTopic
import pandas as pd
import pickle
from random import sample

data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')
doc_processed = list(input_data['abstract'])
doc_processed = sample(doc_processed, int(0.1*len(doc_processed)))


model = BERTopic(nr_topics=19, calculate_probabilities=True, n_gram_range=(1,2), top_n_words=15, verbose=True)
topics, probs = model.fit_transform(doc_processed)

bertopic_file = "models/BERTopic_trained_model.sav"
pickle.dump(model, open(bertopic_file, 'wb'))