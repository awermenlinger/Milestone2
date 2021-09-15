from bertopic import BERTopic
import pandas as pd
import pickle
from random import sample
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from gensim.parsing.preprocessing import STOPWORDS


data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')
doc_processed = list(input_data['abstract'])

addtl_stop_words = ["patient", "patients", "group", "groups" "placebo", "survival", "treatment", "response", "remission",
                     "day", "days", "week", "weeks", "month", "months", "year", "years", "median", "dose", "doses", "result", "results",
                     "conclusion", "conclusions", "study", "significance", "significant", "arm", "arms", "random", "clinical",
                     "trial", "trials", "effect"]
stop_words = STOPWORDS.union(set(addtl_stop_words))



vectorizer_model = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words)
umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine')
model = BERTopic(language="english", calculate_probabilities=False, n_gram_range=(1,2), top_n_words=15,
                 verbose=True, min_topic_size=50, umap_model=umap_model)
topics, probs = model.fit_transform(doc_processed)

bertopic_file = "models/BERTopic_trained_model.sav"
pickle.dump(model, open(bertopic_file, 'wb'))

print(model.get_topic_freq().head())