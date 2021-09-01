import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

RANDOM_SEED = 42

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


df = pd.read_csv('C:/Users/melan/Google Drive/Mike\'s Documents/Projects/Milestone2/data/pubmed_articles_clean.csv')
df['created_date'] = pd.to_datetime(df['created_date'])
df.set_index('created_date', inplace=True)
df.dropna(inplace=True)
df['pubmed_id'] = df['pubmed_id'].astype('int32')

# Clean up keywords. Some have duplicates due to an added '*' or different cases
df['key_words'] = df['key_words'].str.replace("\*", "", regex=True)
df['key_words'] = df['key_words'].str.lower()  # Remove * from all key words and lowercase all words

df['key_words'] = df['key_words'].apply(eval)  # Changes str list to list
df['key_words'] = df['key_words'].apply(lemma_list)  # Lemmatizes all words in the key_words

df['mesh_terms'] = df['mesh_terms'].str.replace("\*", "", regex=True)  # Remove * from all the mesh terms
df['mesh_terms'] = df['mesh_terms'].str.lower()  # Lowercase all mesh terms
df['mesh_terms'] = df['mesh_terms'].apply(eval)  # Changes str list to list

mlb = MultiLabelBinarizer()

# Apply multi-label binarization to key words (like one hot encoding)
mlb.fit_transform(df['key_words'])
label_df = pd.DataFrame(mlb.transform(df['key_words']), columns=mlb.classes_)
y = np.asarray(label_df.values)

full_df = label_df
full_df['abstract'] = df['abstract'].values

# _________________________________________________________On key_words________________________________________________

X = full_df['abstract']

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

forest = RandomForestClassifier(random_state=RANDOM_SEED)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=1)
predicts = multi_target_forest.fit(X_train_tfidf, y_train).predict(X_test_tfidf)
print(classification_report(y_test, predicts))
print(accuracy_score(y_test, predicts))
print(hamming_loss(y_test, predicts))

# ______________________________________________________________________________________________________________________
