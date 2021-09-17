from get_dataframe import get_dfs
from write_results import results_to_txt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer


start = datetime.now()
RANDOM_SEED = 42

df, label_df = get_dfs(pct_of_df=0.02, pct_meshterms=0.02)

print(label_df.shape)


y = np.asarray(label_df.iloc[:, :-3].values)
X = label_df['abstract']

# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)


# Initializing bert
model = SentenceTransformer('bert-base-uncased')
# transforming the data
X_train_bert = model.encode(X_train)
X_test_bert = model.encode(X_test)


svc = SVC(decision_function_shape='ovo', degree=2, kernel='linear', random_state=42)
multiout_svc = MultiOutputClassifier(svc)
clf = multiout_svc.fit(X_train_bert, y_train)
predicts = clf.predict(X_test_bert)

grid_search_results = None

runtime = datetime.now()-start
results_to_txt(clf, y_test, predicts, label_df, 'BERT', runtime, grid_search_results)
