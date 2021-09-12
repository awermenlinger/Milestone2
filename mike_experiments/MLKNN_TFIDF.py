from get_dataframe import get_dfs
from write_results import results_to_txt
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.adapt import MLkNN

start = datetime.now()

RANDOM_SEED = 42

df, label_df = get_dfs(pct_of_df=0.02, pct_meshterms=0.01)


print(label_df.shape)


y = np.asarray(label_df.iloc[:, :-3].values)
X = label_df['abstract']

# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)

# initializing TfidfVectorizer
vetorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
# fitting the tf-idf on the given data
vetorizer.fit(X_train)
# transforming the data
X_train_tfidf = vetorizer.transform(X_train)
X_test_tfidf = vetorizer.transform(X_test)


# svc = SVC(gamma="scale")
# multiout_svc = MultiOutputClassifier(svc)


parameters = {'estimator__k': rainge(2, 5),
              'estimator__s': [0.5, 0.7, 1.0],
              }

mlknn = MLkNN()

multiout_mlknn = MultiOutputClassifier(mlknn, n_jobs=-1)
clf = GridSearchCV(multiout_mlknn, parameters, score='f1_micro')
clf.fit(X_train_tfidf, y_train)
grid_search_results = clf.cv_results_


predicts = clf.predict(X_test_tfidf)

runtime = datetime.now()-start
results_to_txt(clf, y_test, predicts, label_df, 'TFIDF', runtime, grid_search_results)
