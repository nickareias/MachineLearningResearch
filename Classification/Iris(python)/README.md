# Classification on the Iris dataset
https://archive.ics.uci.edu/ml/datasets/Iris 

## Abstract
I will use python with sci-kit learn to pre-process this dataset, and neural networks from Keras to classify.
## Information about Dataset:
##### This dataset contains 150 elements each with 5 attributes.  4 discrete attributes and one text classifier.  There are 3 classes each represented by 50 elements.
 
The results of this experiment will be attached in a spreadsheet.
## Goals:
##### The goal of these experiments will be to maximize the accuracy while minimizing the time taken.  I will gather empirical evidence about the different types of functions and parameters that can be used with the Keras library.

## Preprocessing data:
* Normalized discrete data with min-max method.
* Discretized text classifiers into int values
* Classifiers are later converted into vectors to be used by neural network during classification
## Classification:
* A Neural Network will be used to classify elements from the dataset.  It will be trained on a portion of the data, and tested on a separate portion not seen during training.
* Data is shuffled before being partitioned.  Training data is represented by 3/5 of the data and Testing by 2/5
* Keras library is utilized to be able to switch parameters and functions quickly.  By using Keras (with Theano) training can be computed on the GPU, increasing training speed greatly.
   * During expirementation, different functions of these types are used:
      * Loss function (cost function)
      * Optimizer (learning method)
      * Activation functions
   * Hyper parameters are tested at different values:
      * Learning rate
      * Momentum rate
      * Number of Hidden Neurons
      * Number of Epochs
* For these experiments, the network will be using 1 hidden layer.
## Testing:
While testing, my procedure will follow these steps:
1. Choose loss, optimizer, and activation functions
2. Test many different values of hyper parameters to get maximum accuracy
3. repeat

## Observations:
##### I started my expirement by testing the neural network with stochastic gradient descent, means squared error, and sigmoid activaton functions on the hidden and output layers.  These are the simplest way to use a neural network, and I understand these very well fundamentally.  I wanted to be able to get good results with the methods I understood before testing methods I have not seen yet.
##### While using these more basic methods, I was able to achieve a maximum of around 98% classification accuracy.  The accuracy increased as I increased the number of epochs, and I achieved the best results while using low training rate and high number of epochs.  This is a result of the way that stochastic gradient descent works, because it is reliant on a static learning rate.  I achieved the most time efficient results with a moderate learning rate and low epochs with an accuracy of 96% but a much faster time than the 98% run.
##### After switching to more modern metods: categorical cross entropy, and adam optimizer.  The network didn't get better results.  It got comparable results faster than the other methods, but wasn't able to get a better accuracy.  With these new methods, increasing the epochs didn't seem to increase the accuracy.  After a certain point, increasing the epochs did nothing, and sometimes even reduced the accuracy.  This is most likely because the model experienced overfitting after it got to an optimal point.
##### I then swiched the activation functions to relu and softmax on the hidden and output layers, respectively.  This allowed for a modest increase in the accuracy, yielding 98.33% as the highest.  Still, not much more than what was achieved with the simpler methods.  Again, with these new activation functions, increasing the number of epochs didn't increase the accuracy.  Even with 20 epochs, I was still getting good results comparable to all the other trials.  It was only when I reduced the epochs to 10 that the accuracy started to lower.  This highlights one of the benefits of the adam optimizer; it is very quick to reach it's optimal accuracy.  But at a point the model is limited by the ammount of data that is available.  
##### Because this is such a small dataset (150 elements,) It is likely that there are some outliers that don't line up with the general form of the data.  I believe this is the reason why I was unable to reach 100% accuracy.  Changing the hyper parameters had negligible effects, which suggests that the dataset is very easy to solve, and the final 2-3% that  I couldn't get is due to flaws in the data.