from datetime import datetime
from get_dataframe import get_dfs
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np

start = datetime.now()
RANDOM_SEED = 42
print('getting dataframes')
df, label_df = get_dfs(pct_of_df=0.001, pct_meshterms=0.01)

print(label_df.shape)

print('setting up x and y')
y = np.asarray(label_df.iloc[:, :-3].values)
X = label_df['abstract']

print('setting up BERT')
model = SentenceTransformer('sentence-transformers/allenai-specter')
X_bert = model.encode(X)

print('splitting data into train/test')
# splitting the data to training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.30, random_state=RANDOM_SEED)


X_train, X_test, y_train, y_test = torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(4)
torch.set_num_interop_threads(4)


embeddingSize = 768
hiddenSize = 10
dropoutRate = 0.5
numEpochs = 5
vocabSize = 21
pad = 1
unk = 0


class MyRNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.name = model
        self.LSTM = (model == 'LSTM' or model == 'BiLSTM')
        self.bidir = (model == 'BiLSTM')

        self.embed = nn.Embedding(vocabSize, embeddingSize, padding_idx=pad)

        if model == 'RNN':
            self.rnn = nn.RNN(embeddingSize, hiddenSize)
        elif model == 'GRU':
            self.rnn = nn.GRU(embeddingSize, hiddenSize)
        else:
            self.rnn = nn.LSTM(embeddingSize, hiddenSize, bidirectional=self.bidir)

        self.dense = nn.Linear(hiddenSize * (2 if self.bidir else 1), 1)
        self.dropout = nn.Dropout(dropoutRate)

    def forward(self, text, textLengths):
        embedded = self.dropout(self.embed(text))

        packedEmbedded = nn.utils.rnn.pack_padded_sequence(embedded, textLengths)
        if self.LSTM:
            packedOutput, (hidden, cell) = self.rnn(packedEmbedded)
        else:
            packedOutput, hidden = self.rnn(packedEmbedded)

        output, outputLengths = nn.utils.rnn.pad_packed_sequence(packedOutput)
        if self.bidir:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[0]

        return self.dense(self.dropout(hidden))


basicRNN = MyRNN(model='RNN')
models = [basicRNN]

for model in models:
    if model is None:
        continue
    model.embed.weight.data.copy_(X_train)
    model.embed.weight.data[unk] = torch.zeros(embeddingSize)
    model.embed.weight.data[pad] = torch.zeros(embeddingSize)


criterion = nn.BCEWithLogitsLoss()


def batchAccuracy(preds, targets):
    roundedPreds = (preds >= 0)
    return (roundedPreds == targets).sum().item() / len(preds)


# Training

for model in models:
    if model is not None:
        model.train()

for model in models:
    if model is None:
        continue

    torch.manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(numEpochs):
        epochLoss = 0
        for batch in y_train:
            print(batch.values())
            optimizer.zero_grad()
            text, textLen = batch, batch.size()[0] #[0]
            print(text, textLen)
            predictions = model(text, textLen)#.squeeze(1)
            loss = criterion(predictions, batch[1])
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        print(f'Model: {model.name}, Epoch: {epoch + 1}, Train Loss: {epochLoss / len(y_train)}')
    print()

for model in models:
    if model is not None:
        model.eval()

with torch.no_grad():
    for model in models:

        if model is None:
            continue

        accuracy = 0.0
        for batch in y_test:
            text, textLen = batch[0]
            predictions = model(text, textLen)#.squeeze(1)
            loss = criterion(predictions, batch[1])
            acc = batchAccuracy(predictions, batch[1])
            accuracy += acc
        print('Model: {}, Validation Accuracy: {}%'.format(model.name, accuracy / len(y_test) * 100))

