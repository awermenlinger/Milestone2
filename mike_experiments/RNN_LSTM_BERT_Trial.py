from datetime import datetime
import numpy as np
from write_results import results_to_txt
from get_dataframe import get_dfs
from sentence_transformers import SentenceTransformer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Reference:
# https://towardsdatascience.com/text-classifier-with-multiple-outputs-and-multiple-losses-in-keras-4b7a527eb858

start = datetime.now()
RANDOM_SEED = 42
print('getting dataframes')
df, label_df = get_dfs(pct_of_df=1, pct_meshterms=0.01)

print(label_df.shape)
print('setting up x and y')
y = np.asarray(label_df.iloc[:, :-3].values)
X = label_df['abstract']

print('setting up BERT')
model = SentenceTransformer('sentence-transformers/allenai-specter')
X_bert = model.encode(X)

# import numpy as np
# bert = np.load('C:/Users/melan/Google Drive/Mike\'s Documents/Programming/Jupyter Notebooks/PubMed/bert_sentence_tf_26525_20.npy')
# print(len(bert[:, 0:1]))

MAXLENGTH = 26525
vocabulary_size = 20000
X_bk = pad_sequences(X_bert, MAXLENGTH)

print('splitting data into train/test')
# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.30, random_state=RANDOM_SEED)


main_input = Input(shape=(MAXLENGTH,), dtype='int32', name='main_input')
x = Embedding(input_dim=vocabulary_size, output_dim=50, input_length=MAXLENGTH)(main_input)
x = Dropout(0.3)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=4)(x)
x = LSTM(100)(x)
x = Dropout(0.3)(x)


output_array = []
metrics_array = {}
loss_array = {}

output_columns_binary = label_df.columns[:-3]
output_columns_categorical = ['related']

for i, dense_layer in enumerate(output_columns_binary):
    name = f'binary_output_{i}'
    # A Dense Layer is created for each output
    binary_output = Dense(1, activation='sigmoid', name=name)(x)
    output_array.append(binary_output)
    metrics_array[name] = 'binary_accuracy'
    loss_array[name] = 'binary_crossentropy'


categorical_output = Dense(3, activation='softmax', name='categorical_output')(x)
output_array.append(categorical_output)
metrics_array['categorical_output'] = 'sparse_categorical_accuracy'
loss_array['categorical_output'] = 'sparse_categorical_crossentropy'


model = Model(inputs=main_input, outputs=output_array)


model.compile(optimizer='adadelta',
              loss=loss_array,
              metrics = metrics_array)

y_train_output = []
for col in output_columns_binary:
    y_train_output.append(y_train[col])

for col in output_columns_categorical:
    y_train_output.append(y_train[col])

weight_binary = {0: 0.5, 1: 7}
weight_categorical = {0: 1.4, 1: 0.43, 2: 7}

classes_weights = {}
for i, dense_layer in enumerate(output_columns_binary):
    name = f'binary_output_{i}'
    classes_weights[name] = weight_binary
for i, dense_layer in enumerate(output_columns_categorical):
    name = 'categorical_output'
    classes_weights[name] = weight_categorical


model.fit(X_train, y_train_output,
          epochs=40, batch_size=512,
         class_weight=classes_weights, verbose=0)

y_pred = model.predict(X_test)
THRESHOLD = 0.5 # threshold between classes

f1_score_results = []
# Binary Outputs
for col_idx, col in enumerate(output_columns_binary):
    print(f'{col} accuracy \n')

    # Transform array of probabilities to class: 0 or 1
    y_pred[col_idx][y_pred[col_idx] >= THRESHOLD] = 1
    y_pred[col_idx][y_pred[col_idx] < THRESHOLD] = 0
    f1_score_results.append(f1_score(y_test[col], y_pred[col_idx], average='macro'))
    print(classification_report(y_test[col], y_pred[col_idx]))

# Multi Class Output
for col_idx, col in enumerate(output_columns_categorical):
    print(f'{col} accuracy \n')

    # Select class with higher probability from the softmax output: 0, 1 or 2
    y_pred_2 = np.argmax(y_pred[-1], axis=-1)
    f1_score_results.append(f1_score(y_test[col], y_pred_2, average='macro'))
    print(classification_report(y_test[col], y_pred_2))
print('Total :', np.sum(f1_score_results))
