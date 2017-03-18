# Classification on the Adult dataset
https://archive.ics.uci.edu/ml/datasets/Adult

## Abstract:
I will use a neural network to classify this dataset.  The data is already split into training and testing data, but because of misrepresentation in the testing data, I will mix the two sets together and re-split them myself.  I will expirement on various architectures and parameter configurations.  I will explain the process I use to decide on these parameters and architecture.  Relevant findings will be included.
## Preprocessing:
* There are many categorical attributes in this dataset, so I will have to develop a strategy to convert them into a form that can be used with a neural network.
* There is a categorical attribute for education, but also a number value for education.  I have considered omiting the categorical education variable.  Keeping it (in addition to the continuous education variable) should give more accuracy to the model because it will have more accurate information about the differences in education level.
* Other than this, there are 7 other categorical attributes.  For each of these I will use one-hot encoding to create dummy variables for each level of each attribute.
* All continuous attributes will be min-max normalized
* Class will be converted into an int value of (0, 1) corresponding to <=50k and >50k respectively.
* 13% of the training data included at least one missing attribute, and some included more.
* In the training data provided, one of the countries is not represented.  To remedy this, I mixed the data together to pre-process it all at once, splitting into training/testing will be done before classification.  A cross validation split will be utilized to ensure fair splitting of the data.
* For the country data, 91% of it is from the united states.  Because of this, I will try encoding country with either US, or foreign.  I will also change any missing country elements to foreign.
* The training data also had periods after each of the classifiers, while the testing data did not.  This made it so that there were 4 classifiers when there should have been 2.  I had to remove the periods before compiling the data.
* After preliminary tests I tried different methods of converting categorical data to numerical.  They are stored in separate files:
   * adult_onehot – one hot encoding for all categorical data, no attributes removed
   * adult_integer – integer conversion for all categorical data, no attributes removed
   * adult_onehot_2_countries – onehot encoding, but countries are represented as either US or foreign.
   * adult_onehot_2_countries_no_edu – same as previous, but continuous education attribute is removed.  This should put more weight in the one-hot encoded categorical education attribute which should lead to more accurate results.  The number for education doesn't hold the correct information.  The difference between a high school graduate (9) and someone with their bachelors (13) is only 4.  this should probably be a much higher number.  The one-hot encoding for education will preserve the clear distinction between different levels of education.


## Observations:
##### After some preliminary trials I have gotten a classification accuracy of around 85% on the test set.  These are decent results, but not as good as I hoped for.  My first insticts tell me that the remaining 15% has something to do with the way the categorical data was translated into numbers.  But I will try some different values for hyper parameters with this version of the data to see how much accuracy we can get out of it.	
##### Classification accuracy starts at a reasonable number, and rises slowly and steadily.  The accuracies get erratic around the higher epoch ranges.  This is likely due to the learning rate being set too high.  
![30 epochs](graphs/sgd_onehot_30_zoom.png?raw=true) ![100 epochs](graphs/sgd_onehot_100_zoom.png?raw=true)
##### Testing with a lower learning rate and higher number of epochs gave me a more smooth increase in accuracy.  The drawback to this is that it didn't get much better results, and it began taking exponentially longer.  In the experiment with 1600 epochs, the increase in accuracy virtually stops once it gets to around 900 epochs.  In all 4 of these experiments the accuracy got to around 85%.  This supports my original belief that the inaccuracy is caused by how the data was preprocessed
![400 epochs](graphs/sgd_onehot_400_low_lr.png?raw=true) ![1600 epochs](graphs/sgd_onehot_1600_low_lr.png?raw=true)
##### I went back to pre-processing and created multiple different versions of the dataset to test.  I also implemented cross validation to reduce oscilations in accuracy.  I cross validate by splitting the dataset N times, using each split as the testing data once, then averaging results across all the trials.  There are still oscilations, but this helped to even out variations due to random chance.  
##### After trying a few different methods of converting categorical to numerical, the results of each method are extremely similar..  Converting to integers rather than one-hot encoding worsened the results, showing me that overall, one-hot encoding is better.  At this point, running this program is starting to take a long time, almost 12 minutes each time.  This is because to cross validate with splits of 1/3 it has to classify each dataset 3 times.  And with 50,000 elements per data set run through 100 epochs, runtimes start to get very long.  Judging from this graph however, it seems that cutting the epochs around 40 or 50 would be a good choice if I was trying to save time.  The increase after that point seems negligible.  After looking at the graph of 200 epochs, it seems like the purple line (the most up-to-date processing) has a slight advantage.  Also, after 100 epochs it seems like there is virtually no increase in accuracy.
![4 files 100](graphs/sgd_4types_100.png?raw=true) ![4 files 200](graphs/sgd_4types_200.png?raw=true)
##### In the end it seems like one hot encoding is the way to go (over integer encoding) but other than that, the data preparation didn't seem to change the outcome very much.  It must be that the attributes I changed didn't have much to do with the classifer.  This is actually good news because I was able to take out some of the unnecessary data from the set without compromising results. 






















The next thing I would like to try is to find a different way of converting the categorical data into numerical.  Comparing these different methods might give some empirical evidence about the choices.
	I would also like to expeiment with different structures of the neural network.  Adding dropout layers might yield some extra accuracy.