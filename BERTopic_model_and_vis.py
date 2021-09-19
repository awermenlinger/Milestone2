## Works, but no coherence score

from bertopic import BERTopic
import pandas as pd
import pickle
from random import sample
from umap import UMAP


data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')
doc_processed = list(input_data['abstract'])
#print (len(doc_processed))

#optimized with the FAQ: https://maartengr.github.io/BERTopic/faq.html#i-have-only-a-few-topics-how-do-i-increase-them
umap_model = UMAP(n_neighbors=200, n_components=10, min_dist=0.0, metric='cosine')
model = BERTopic(language="english", calculate_probabilities=False, n_gram_range=(1,2), top_n_words=15,
#model = BERTopic(language="english", calculate_probabilities=False, n_gram_range=(1,1), top_n_words=15,
                 verbose=True, min_topic_size=300, umap_model=umap_model)
topics, probs = model.fit_transform(doc_processed)

bertopic_file = "models/BERTopic_trained_model_nobigrams.sav"
model.save(bertopic_file)

print(model.get_topic_info())