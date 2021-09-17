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

# svc = SVC(decision_function_shape='ovo', degree=2, kernel='linear', random_state=42)
# multiout_svc = MultiOutputClassifier(svc)
# clf = multiout_svc.fit(X_train, y_train)
# predicts = clf.predict(X_test)


knn = KNeighborsClassifier(n_neighbors=6, weights='distance')

multiout_knn = MultiOutputClassifier(knn)
clf = multiout_knn.fit(X_train, y_train)
predicts = clf.predict(X_test)

grid_search_results = None

runtime = datetime.now()-start
results_to_txt(clf, y_test, predicts, label_df, 'BERT', runtime, grid_search_results)
