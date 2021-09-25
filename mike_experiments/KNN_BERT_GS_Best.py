from get_dataframe import get_dfs
from write_results import results_to_txt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer, models
from nltk.stem import WordNetLemmatizer
import pickle

start = datetime.now()
RANDOM_SEED = 42
print('getting dataframes')
df, label_df = get_dfs(pct_of_df=1, pct_meshterms=0.01)

print(label_df.shape)

print('setting up x and y')
y = np.asarray(label_df.iloc[:, :-3].values)
# X = label_df['abstract']


# def lemmatize(lst):
#     lemmatizer = WordNetLemmatizer()
#     return " ".join([lemmatizer.lemmatize(x) for x in lst.split()])
#
#
# X_lemmatized = X.apply(lemmatize)


# print('setting up BERT')
# model = SentenceTransformer('sentence-transformers/allenai-specter')
# X_bert = model.encode(X_lemmatized)

X_bert = np.load('results/bert_sentence_tf30328_3_1_001.npy')

print('splitting data into train/test')
# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.30, random_state=RANDOM_SEED)


# Best from gridsearch
# {'estimator__algorithm': 'auto',
# 'estimator__leaf_size': 15,
# 'estimator__metric': 'euclidean',
# 'estimator__n_neighbors': 5,
# 'estimator__p': 1,
# 'estimator__weights': 'uniform'}
knn = KNeighborsClassifier(algorithm='auto',
                           leaf_size=15,
                           metric='euclidean',
                           n_neighbors=5,
                           p=1,
                           weights='uniform',
                           n_jobs=6
                           )

multiout_knn = MultiOutputClassifier(knn)
clf = multiout_knn.fit(X_train, y_train)
predicts = clf.predict(X_test)

grid_search_results = None


pickle.dump(clf, open('results/knn_bert_model.sav'))

runtime = datetime.now()-start
results_to_txt(clf, y_test, predicts, label_df, 'BERT', runtime, grid_search_results)
