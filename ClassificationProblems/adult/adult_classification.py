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

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

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

    return data

def vectorized_int(num, size):
    classifier = np.zeros([size])
    for i in range(size):
        if(i == num):
            classifier[i] = 1
                      
    return classifier

def train_net(training_data, testing_data, epochs):

    #training_data, testing_data = get_data(path)    
    
    #gets everything but last column (classifier) and store as numpy array
    x_train = np.array(transpose(transpose(training_data)[:-1]))
    x_test = np.array(transpose(transpose(testing_data)[:-1]))
    
    #gets only last column (classifier) and store as numpy aray
    y_train = transpose(training_data)[-1]
    y_test = transpose(testing_data)[-1]
    
    #convert int classifier to a vector 
    y_train = np.array([vectorized_int(x, 2) for x in y_train])
    y_test = np.array([vectorized_int(x, 2) for x in y_test])
    
    num_inputs = np.shape(x_train)[1]
    hidden_neurons = 100
    num_classes = 2
    
    #epochs = 100
    batch_size = 50
    
    learning_rate = 0.1
    mom = 0.1
    
    # create model
    print("creating model")
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=num_inputs, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='sigmoid'))
    
    sgd = optimizers.SGD(lr=learning_rate, momentum=mom, decay=0.0, nesterov=False)
    
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    start = time()
    
    # Fit the model
    print("training model...")
    results = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch_size, verbose=2)
    
    end = time()


    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    scores2 = model.evaluate(x_train, y_train, verbose=0)
    print("\nAccuracy: %.2f%%" % ((scores[1] + scores2[1]) / 2*100))
    print("Time: %.2f" % (end - start))

    return results


#splits data into multiple parts and cross validates them
#runs algorithm num_folds times with each fold as the test
#data once
def cross_validate(path, num_folds, epochs):
    
    data = get_data(path)
    folds = cross_validation_split(data, num_folds)
    results = list()
    
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
                    
        results.append(train_net(train_set, test_set, epochs))

    return results

def get_avg_accuracies(results):
                      
    val_accuracies = []
    for i in results:
        val_accuracies.append(i.history['val_acc'])
    
    avg_accuracies = [np.mean(x) for x in transpose(val_accuracies)]
    
    return avg_accuracies






##########
## MAIN ##
##########


epochs = 200

"""
results1 = train_net("adult_onehot.txt", epochs)
results2 = train_net("adult_integer.txt", epochs)
results3 = train_net("adult_onehot_2_countries.txt", epochs)
results4 = train_net("adult_onehot_2_countries_no_edu.txt", epochs)
"""


results1 = cross_validate("adult_onehot.txt", 3, epochs)
avg_max_results1 = np.mean([max(results1[i].history['val_acc']) for i in range(len(results1))])
avg_acc1 = get_avg_accuracies(results1)


results2 = cross_validate("adult_integer.txt", 3, epochs)
avg_max_results2 = np.mean([max(results2[i].history['val_acc']) for i in range(len(results2))])
avg_acc2 = get_avg_accuracies(results2)

results3 = cross_validate("adult_onehot_2_countries.txt", 3, epochs)
avg_max_results3 = np.mean([max(results3[i].history['val_acc']) for i in range(len(results3))])
avg_acc3 = get_avg_accuracies(results3)


results4 = cross_validate("adult_onehot_2_countries_no_edu.txt", 3, epochs)
avg_max_results4 = np.mean([max(results4[i].history['val_acc']) for i in range(len(results4))])
avg_acc4 = get_avg_accuracies(results4)

e = np.arange(0, epochs, 1)

colors = ['teal', 'yellowgreen', 'gold', 'darkviolet']
lw = 2

plt.plot(e, avg_acc1, color = colors[0], label = "OH", linewidth = lw)
plt.plot(e, avg_acc2, color = colors[1], label = "Integer", linewidth = lw)
plt.plot(e, avg_acc3, color = colors[2], label = "OH US/Foreign", linewidth = lw)
plt.plot(e, avg_acc4, color = colors[3], label = "OH US/Foreign\nno edu #", linewidth = lw)

plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title('SGD '+str(epochs)+" Epochs")

plt.show()

maxes = [avg_max_results1,avg_max_results2,avg_max_results3,avg_max_results4]
width = 0.9
ind = np.arange(len(maxes))
plt.bar(ind,maxes,width)
axes = plt.gca()
axes.set_ylim([0.85,0.86])
plt.show()
"""
axes = plt.gca()
axes.set_xlim([-1,epochs])
axes.set_ylim([0.82,0.86])
"""

#plt.show()