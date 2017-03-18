import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
from time import time

seed = 9
np.random.seed(seed)

def transpose(data):
    return [[row[i] for row in data] for i in range(len(data[0]))] 

def get_data(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    
    #get data from lines
    data = []
    for line in lines:
        temp_row = []
        temp_element = ""
        for char in line:
            if(char != ',' and char != '\n'):
                temp_element += char
            else:
                try:
                    temp_row.append(float(temp_element))
                except:
                    temp_row.append(temp_element)
                temp_element = ""
                
        data.append(temp_row)
        
    #put 4/5 of the data into training
    #and 1/5 into testing
    l = int(len(data) * (3/5))
    
    random.shuffle(data)

    training_data = data[:l]
    testing_data = data[l:]
        
    return (training_data, testing_data)

def vectorized_int(num):
    classifier = np.zeros([3])
    for i in range(3):
        if(i == num):
            classifier[i] = 1
                      
    return classifier

training_data, testing_data = get_data("data/iris_data_preprocessed.txt")

#gets everything but last column (classifier) and store as numpy array
x_train = np.array(transpose(transpose(training_data)[:-1]))
x_test = np.array(transpose(transpose(testing_data)[:-1]))

#gets only last column (classifier) and store as numpy aray
y_train = transpose(training_data)[-1]
y_test = transpose(testing_data)[-1]

#convert int classifier to a vector 
y_train = np.array([vectorized_int(x) for x in y_train])
y_test = np.array([vectorized_int(x) for x in y_test])


#network structure variables
#hard coded based on dataset
num_inputs = 4
hidden_neurons = 30
num_classes = 3

#hyper parameters
epochs = 30
batch_size = 1
learning_rate = 0.15
mom = 0.15

# create model
print("creating model")
model = Sequential()
model.add(Dense(hidden_neurons, input_dim=num_inputs, init='normal', activation='relu'))
model.add(Dense(num_classes, init='normal', activation='softmax'))


# Compile model

#using sgd
sgd = optimizers.SGD(lr=learning_rate, momentum=mom, decay=0.0, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time()

# Fit the model
print("training model...")
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch_size, verbose=2)

end = time()


# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
scores2 = model.evaluate(x_train, y_train, verbose=0)
print("\nAccuracy: %.2f%%" % ((scores[1] + scores2[1]) / 2*100))
print("Time: %.2f" % (end - start))


