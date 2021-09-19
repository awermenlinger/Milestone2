import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_SEED = 42


def concat_article_files(pct_of_df):
    files = ['pubmed_articles_cancer_01_smaller.csv',
             'pubmed_articles_cancer_02_smaller.csv',
             'pubmed_articles_cancer_03_smaller.csv',
             'pubmed_articles_cancer_04_smaller.csv'
             ]

    file_path = '../data/'
    dfs = pd.DataFrame()

    for file in files:
        df = pd.read_csv(f"{file_path}{file}", low_memory=False).dropna(how='all')
        dfs = pd.concat([dfs, df])

    return dfs.sample(frac=pct_of_df, random_state=RANDOM_SEED)


def lemma_list(lst):
    lemmatizer = WordNetLemmatizer()
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
    df = df.where(df['created_date'].dt.year >= 2015).dropna(how='all')
    df.set_index('created_date', inplace=True)
    df['pubmed_id'] = df['pubmed_id'].astype('int32')

    df['mesh_terms'] = df['mesh_terms'].str.replace("\*", "", regex=True)  # Remove * from all the mesh terms
    df['mesh_terms'] = df['mesh_terms'].str.lower()  # Lowercase all mesh terms
    df['mesh_terms'] = df['mesh_terms'].apply(eval)  # Changes str list to list
    df['mesh_terms'] = df['mesh_terms'].apply(lemma_list)

    return df[['pubmed_id', 'mesh_terms', 'abstract']]


def multilabel_binerized(df, pct_meshterms):
    mlb = MultiLabelBinarizer()

    # Apply multi-label binarization to key words (like one hot encoding)
    label_df = pd.DataFrame(mlb.fit_transform(df['mesh_terms']).astype('int8'), columns=mlb.classes_)
    label_df.drop([col for col, val in label_df.sum().iteritems() if val < (label_df.shape[0] * pct_meshterms)],
                  axis=1,
                  inplace=True
                  )

    label_df.dropna(how='all', inplace=True)
    label_df['abstract'] = df['abstract'].values
    label_df['created_date'] = df.index
    label_df['pubmed_id'] = df['pubmed_id'].values

    return label_df


def get_dfs(pct_of_df, pct_meshterms):
    df = concat_article_files(pct_of_df)
    df = clean_df(df)
    label_df = multilabel_binerized(df, pct_meshterms)
    return df, label_df
