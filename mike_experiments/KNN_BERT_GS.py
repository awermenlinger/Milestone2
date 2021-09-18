from get_dataframe import get_dfs
from write_results import results_to_txt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer, models


start = datetime.now()
RANDOM_SEED = 42

df, label_df = get_dfs(pct_of_df=0.01, pct_meshterms=0.02)

print(label_df.shape)


y = np.asarray(label_df.iloc[:, :-3].values)
X = label_df['abstract']

model = SentenceTransformer('sentence-transformers/allenai-specter')
X_bert = model.encode(X)


# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.30, random_state=RANDOM_SEED)


# knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
#
# multiout_knn = MultiOutputClassifier(knn)
# clf = multiout_knn.fit(X_train, y_train)
# predicts = clf.predict(X_test)
#
# grid_search_results = None

parameters = {'estimator__n_neighbors': range(3, 7),
              'estimator__weights': ['uniform', 'distance'],
              'estimator__algorithm': ['auto', 'kd_tree'],
              'estimator__leaf_size': [15, 30, 45],
              'estimator__metric': ['euclidean',
                                    'minkowski',
                                    'hamming',
                                    'jaccard'
                                    ],
              'estimator__p': [1, 2, 3],
              }

knn = KNeighborsClassifier()

multiout_knn = MultiOutputClassifier(knn)
clf = GridSearchCV(multiout_knn, parameters, scoring='f1_micro', n_jobs=6)
clf.fit(X_train, y_train)
grid_search_results = clf.cv_results_

predicts = clf.predict(X_test)

runtime = datetime.now()-start
results_to_txt(clf, y_test, predicts, label_df, 'BERT', runtime, grid_search_results)
