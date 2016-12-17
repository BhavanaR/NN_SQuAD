import numpy as np
from keras.preprocessing import sequence
import json
from squad_dataset import get_squad_data
from keras.models import load_model
X,Y,vocab = get_squad_data()
X = X[100:200]
Y = Y[100:200]

with open('IntToWordMap.json','r') as file:
    int_to_word = json.load(file)

word_to_int = dict((word, i) for i, word in int_to_word.items())

seq_length = 30
ans_seq_length = 5

dataX = []
for i in range(0, len(X)):
	dataX.append([word_to_int[word] for word in X[i]])

dataX = np.array(sequence.pad_sequences(dataX,maxlen=seq_length))

for i in range(1,2):
    model = load_model('squad_linear_{0}_epoch.h5'.format(i))
    print(model.summary())
    print("Loaded model")
    print("Predicting")
    predictedY = model.predict(dataX[:1,:])
    predictedWords = []
    #for i in range(0,len(Y)+1):
    predictedWords.append([int_to_word[str(abs(int(num)))] for num in predictedY[0,:] if int(num) != 0])
    print("Answers are:")
    #for i in range(0,len(Y)+1):
    print("Context:")
    print(X[0])
    print("Answer:")
    print(Y[0])
    print("Predicted:")
    print(predictedWords[0])

