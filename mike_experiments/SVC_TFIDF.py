from get_dataframe import get_dfs
from write_results import results_to_txt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, classification_report

RANDOM_SEED = 42

df, label_df = get_dfs(pct_of_df=0.02, pct_meshterms=0.05)

print(df.shape)
print(label_df.shape)
# print(label_df)

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


svc = SVC(gamma="scale")
multiout_svc = MultiOutputClassifier(svc)

# multiout_svc = MultiOutputClassifier(estimator=SVC(C=1.0, break_ties=False, cache_size=200,
#                                     class_weight=None, coef0=0.0,
#                                     decision_function_shape='ovr', degree=3,
#                                     gamma='scale', kernel='rbf', max_iter=-1,
#                                     probability=False, random_state=None,
#                                     shrinking=True, tol=0.001, verbose=False),
#                       n_jobs=-1)

clf = multiout_svc.fit(X_train_tfidf, y_train)

predicts = clf.predict(X_test_tfidf)
# print(classification_report(y_test, predicts))

results_to_txt(clf, y_test, predicts, label_df, 'TFIDF')

# estimator = clf.get_params()
# accuracy = accuracy_score(y_test, predicts)
# f1_weighted = f1_score(y_test, predicts, average='weighted')
# f1_micro = f1_score(y_test, predicts, average='micro')
# hammingLoss = hamming_loss(y_test, predicts)
# precision_avg_samples = precision_score(y_test, predicts, average='samples')
# class_report = classification_report(y_test, predicts)
#
# txt_body = f"{estimator}\nDataframe Size: {label_df.size}\n\nAccuracy: {accuracy}\nF1 Score (weighted): {f1_weighted}\n\
# F1 Score (micro): {f1_micro}\nHamming Loss: {hammingLoss}\nPrecision (average by samples): \nClassification Report: \n\
# {class_report} "
#
# print(txt_body)
# filepath = "/results"
# filename = f"{estimator['estimator'][:-2]}_tfidf.txt"
#
# with open(filename) as file:
#     file.write(txt_body)



# print(f"Accuracy: {accuracy_score(y_test, predicts)}")
# print(f"F1 Score (weighted): {f1_score(y_test, predicts, average='weighted')}")
# print(f"F1 Score (micro): {f1_score(y_test, predicts, average='micro')}")
# print(f"Hamming Loss: {hamming_loss(y_test, predicts)}")
# print(f"Precision (average by samples): {precision_score(y_test, predicts, average='samples')}")
