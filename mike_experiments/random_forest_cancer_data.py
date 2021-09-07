import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss, accuracy_score


def concat_article_files():
    files = ['pubmed_articles_cancer_01.csv',
             'pubmed_articles_cancer_02.csv',
             'pubmed_articles_cancer_03.csv',
             'pubmed_articles_cancer_04.csv'
             ]

    file_path = '../../../Projects/Milestone2/data/'
    dfs = pd.DataFrame()

    for file in files:
        df = pd.read_csv(f"{file_path}{file}", low_memory=False).dropna(how='all')
        dfs = pd.concat([dfs, df])

    return dfs


lemmatizer = WordNetLemmatizer()


def lemma_list(lst):
    stemmed = []
    for key_word in lst:
        new_key_word = ""
        if len(key_word.split()) >= 2:
            for word in key_word.split():
                new_key_word += lemmatizer.lemmatize(word) + " "

        else:
            new_key_word = lemmatizer.lemmatize(key_word)

        stemmed.append(new_key_word.strip())

    return stemmed


def clean_df(df):
    df = df.loc[(~df['mesh_terms'].isnull()) & (~df['abstract'].isnull())].copy()
    df['created_date'] = pd.to_datetime(df['created_date'])
    df.set_index('created_date', inplace=True)
    #     df.dropna(inplace=True)
    df['pubmed_id'] = df['pubmed_id'].astype('int32')

    df['mesh_terms'] = df['mesh_terms'].str.replace("\*", "", regex=True)  # Remove * from all the mesh terms
    df['mesh_terms'] = df['mesh_terms'].str.lower()  # Lowercase all mesh terms
    df['mesh_terms'] = df['mesh_terms'].apply(eval)  # Changes str list to list
    df['mesh_terms'] = df['mesh_terms'].apply(lemma_list)


    return df[['pubmed_id', 'mesh_terms', 'abstract']]

df = concat_article_files()
df = clean_df(df)

mlb = MultiLabelBinarizer()

# Apply multi-label binarization to key words (like one hot encoding)
mlb.fit_transform(df['mesh_terms'])
label_df = pd.DataFrame(mlb.fit_transform(df['mesh_terms']).astype('int8'), columns=mlb.classes_)
label_df
label_df = label_df[label_df.columns[label_df.sum()>1]]

label_df['abstract'] = df['abstract'].values
label_df['created_date'] = df.index
label_df['pubmed_id'] = df['pubmed_id'].values


y = np.asarray(label_df.values)
X = label_df['abstract']

RANDOM_SEED = 42

# initializing TfidfVectorizer
vetorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)

# fitting the tf-idf on the given data
vetorizer.fit(X_train)
vetorizer.fit(X_test)
# transforming the data
X_train_tfidf = vetorizer.transform(X_train)
X_test_tfidf = vetorizer.transform(X_test)

forest = RandomForestClassifier(random_state=RANDOM_SEED, verbose=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
clf = multi_target_forest.fit(X_train_tfidf, y_train)
predicts = clf.predict(X_test_tfidf)
print(classification_report(y_test, predicts))
print(accuracy_score(y_test, predicts))
print(hamming_loss(y_test, predicts))


