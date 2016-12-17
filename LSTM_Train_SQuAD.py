import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from squad_dataset import get_squad_data
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dropout,RepeatVector,Merge
from keras.callbacks import EarlyStopping
import json

def train_test_split(X,Xq,Y, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(len(X) * (1 - test_size))

    X_train = X[:ntrn,:]
    Xq_train = Xq[:ntrn,:]
    Y_train = Y[:ntrn,:]
    
    X_test = X[ntrn:,:]
    Xq_test = Xq[ntrn:,:]
    Y_test = Y[ntrn:,:]
    #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return (X_train, Xq_train, Y_train), (X_test, Xq_test, Y_test)

X,Xq,Y,vocab = get_squad_data(size=2000)

word_to_int = dict((word, i+1) for i, word in enumerate(vocab))
int_to_word = dict((i, word) for word, i in word_to_int.items())

with open('IntToWordMap.json','w') as file:
    json.dump(int_to_word,file)

n_vocab = len(vocab)
print("Total Vocab: ", n_vocab)

context_length = len(max(X,key=len))
ques_length = len(max(Xq,key=len))
ans_length = len(max(Y,key=len))

#print(context_length,ans_length)
#print(X[0],Y[0])
# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataXq = []
dataY = []
for i in range(0, len(X)):
    dataX.append([word_to_int[word] for word in X[i]])
    dataXq.append([word_to_int[word] for word in Xq[i]])
    
    answer_indexes = [word_to_int[word] for word in Y[i]]  
    #print(answer_indexes,Y[i])
    Y_vector = np.zeros(len(vocab)+1,dtype=np.uint32)
    Y_vector[answer_indexes] = 1
    dataY.append(Y_vector)
    
dataX = np.array(sequence.pad_sequences(dataX,maxlen=context_length),dtype=np.uint32)
dataXq = np.array(sequence.pad_sequences(dataXq,maxlen=ques_length),dtype=np.uint32)
dataY = np.array(dataY,dtype=np.uint32)

print("DataX shape : {0}, DataXq shape : {1}, DataY shape : {2}".format(dataX.shape,dataXq.shape,dataY.shape))
#print("DataX sample: {0},DataY sample : {1}".format(dataX[0],dataY[0]))

test_size = 0.2
(X_train, Xq_train, Y_train) , (X_test, Xq_test, Y_test) =  train_test_split(dataX,dataXq,dataY,test_size)
print("X_train shape : {0}, Xq_train shape : {1}, Y_train shape : {2}".format(X_train.shape,Xq_train.shape,Y_train.shape))
print("X_test shape : {0}, Xq_test shape : {1}, Y_test shape : {2}".format(X_test.shape,Xq_test.shape,Y_test.shape))

# define the LSTM model
embedding_vector_length = 32
epochs = 100
batch_size = 64
embed_hidden_size = 150

sentrnn = Sequential()
sentrnn.add(Embedding(len(vocab)+1, embed_hidden_size,
                      input_length=context_length))
sentrnn.add(Dropout(0.3))

qrnn = Sequential()
qrnn.add(Embedding(len(vocab)+1, embed_hidden_size,
                   input_length=ques_length))
qrnn.add(Dropout(0.3))
qrnn.add(LSTM(embed_hidden_size, return_sequences=False))
qrnn.add(RepeatVector(context_length))

model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='sum'))
model.add(LSTM(embed_hidden_size, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(len(vocab)+1, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
#print (model.summary())

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0, mode='auto')]

print("Epoch : {0}, Neurons : {1}, Batch_size : {2}, ContextLength : {3}, AnswerLength : {4}".format(epochs,embed_hidden_size,batch_size,context_length,ans_length))

model.fit([X_train,Xq_train], Y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.1,callbacks=callbacks)

#model.save('squad_linear_{0}_epoch.h5'.format(epochs))

#rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))
#print("RMSE : ",rmse)

scores = model.evaluate([X_test,Xq_test],Y_test, verbose=1)
print("Accuracy: ",scores )
print("Accuracy: %.2f%%" % (scores[1]*100))

predictedY = model.predict([X_test,Xq_test])

print("Predictions:")
#len(predictedY)
for i in range(0,3):
    #print(predictedY[i])
    sorted_values = predictedY[i].argsort()[-3:][::-1]
    #answer_indexes = np.nonzero(predictedY[i])[0]
    answer_indexes = sorted_values
    print(answer_indexes)
    predictedWords = []
    predictedWords = [int_to_word[int(num)] for num in answer_indexes if num != 0]
    
    if(len(predictedWords)>0):
        print("Answers are:")
        #for i in range(0,len(Y)+1):
        print("Context:")
        print(X[i])
        print("Answer:")
        print(Y[i])
        print("Predicted:")
        print(predictedWords)
    
print("Done")