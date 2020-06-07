# Introduction
K-Nearest Neighbors (KNN) is one of the simplest algorithms used in Machine Learning for regression and classification problem. KNN algorithms use data and classify new data points based on similarity measures (e.g. distance function).
Classification is done by a majority vote to its neighbors. The data is assigned to the class which has the nearest neighbors. As you increase the number of nearest neighbors, the value of k, accuracy might increase.

# Independent Variables
This is a logistic regression model to identify correlations between the following 9 independent variables and the class of the tumor (benign or malignant).
1. Clump thickness
2. Uniformity of cell size
3. Uniformity of cell shape
4. Marginal adhesion
5. Single epithelial cell
6. Bare Nuclei
7. Bland chromatin
8. Normal nucleoli
8. Mitoses

KNeighborsClassifier can identify important predictors of breast cancer using odds ratios and generate confidence intervals that provide additional information for decision-making. Model performance depends on the ability of the radiologists to accurately identify findings on mammograms.

# Overview

To read about different modules refer to the [scikit-learn](https://scikit-learn.org/stable/index.html) site.
### Part 1: Data Preprocessing
##### 1. Importing the dataset
Imported the pandas library to read the dataset. The given dataset is multivariate defined over 10 different attributes. Each attribute is an integer.

There are two class of the tumor: Benign (not breast cancer) represented as 2 or Malignant (breast cancer) represented as 4 based off its characteristics in the dataset.

##### 2. Splitting the dataset into a training set and test set
The dataset was splitted using the test_train_split function imported from model_selection.

Out of Approx. 700 instances, 25% were splitted into test set and remaining 75% were kept to train the dataset called as X_train, y_train, X_test, y_test.
    
### Part 2: Training and Inference
##### 1. Training the logistic regression model on the training set
Trained the KNeighborsClassifier to fit on the splitted X_train and y_train dataset.

##### 2. Predicting the test results
The trained classifier was used to predict the values in test set.

### Part 3: Evaluating the model
##### 1. Making the confusion matrix
The confusion matrix was created to evaluate the model depicting the approx percentage of the correct predicted value comparing from the given test set values. 

##### 2. Computing the accuracy with k-fold cross-validation
Calculated the Accuracy (97.46% approx) and Standard Deviation (2.14%).
