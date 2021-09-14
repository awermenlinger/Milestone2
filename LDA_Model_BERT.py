import pickle
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd


# SETTINGS FOR MODEL
umap_embeddings_file = "models/BERT_trained_umap_embeddings.sav"
umap_data_file = "models/BERT_trained_umap_data.sav"
result_file = "results/BERT_results.sav"

data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')
doc_processed = list(input_data['abstract'])

model = SentenceTransformer('all-mpnet-base-v2') #microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
embeddings = model.encode(doc_processed, show_progress_bar=True, device="cuda")

umap_embeddings = umap.UMAP(n_neighbors=15,
                            metric='cosine').fit_transform(embeddings)
pickle.dump(umap_embeddings, open(umap_embeddings_file, 'wb'))

cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
pickle.dump(umap_data, open(umap_data_file, 'wb'))

result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_
pickle.dump(result, open(result_file, 'wb'))

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()


#https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

# @misc{pubmedbert,
#   author = {Yu Gu and Robert Tinn and Hao Cheng and Michael Lucas and Naoto Usuyama and Xiaodong Liu and Tristan Naumann and Jianfeng Gao and Hoifung Poon},
#   title = {Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing},
#   year = {2020},
#   eprint = {arXiv:2007.15779},
# }