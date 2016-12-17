import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from squad_dataset import get_squad_data
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
import json
def train_test_split(X,Y, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(len(X) * (1 - test_size))

    X_train = X[:ntrn,:]
    Y_train = Y[:ntrn,:]
    X_test = X[ntrn:,:]
    Y_test = Y[ntrn:,:]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return (X_train, Y_train), (X_test, Y_test)

X,Y,vocab = get_squad_data()
X = X[:10000]
Y = Y[:10000]

word_to_int = dict((word, i+1) for i, word in enumerate(vocab))
int_to_word = dict((i, word) for word, i in word_to_int.items())

with open('IntToWordMap.json','w') as file:
    json.dump(int_to_word,file)

n_vocab = len(vocab)
print("Total Vocab: ", n_vocab)

seq_length = len(max(X,key=len))
ans_seq_length = len(max(Y,key=len))

print(seq_length,ans_seq_length)
#print(X[0],Y[0])
# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, len(X)):
	dataX.append([word_to_int[word] for word in X[i]])
	dataY.append([word_to_int[word] for word in Y[i]])

dataX = np.array(sequence.pad_sequences(dataX,maxlen=seq_length))
dataY = np.array(sequence.pad_sequences(dataY,maxlen=ans_seq_length))
#print(dataX[0],dataY[0])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

test_size = 0.2
(X_train, Y_train) , (X_test, Y_test) =  train_test_split(dataX,dataY,test_size)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# define the LSTM model
hidden_neurons = 100

embedding_vector_length = 32

#How to give a different dimensions for in and out
model = Sequential()
model.add(Embedding(n_vocab,embedding_vector_length, input_length=seq_length))
model.add(LSTM(hidden_neurons, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(ans_seq_length, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
print (model.summary())

epochs = 100
batch_size = 50
print("Epoch : {0}, Neurons : {1}, Batch_size : {2}, ContextLength : {3}, AnswerLength : {4}".format(epochs,hidden_neurons,batch_size,seq_length,ans_seq_length))

print("Running {0} epochs".format(epochs))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test,Y_test))

model.save('squad_linear_{0}_epoch.h5'.format(i))

predicted = model.predict(X_test)
#rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))
#print("RMSE : ",rmse)

scores = model.evaluate(X_test,Y_test, verbose=0)
print("Accuracy: ",scores )
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Done")