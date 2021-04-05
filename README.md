# Ensemble-Learning

## Data Set
* We will use the Blood Transfusion Service Center Data Set from the UC Irvine Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center

We have split the above data into training (**train.csv**) and testing (**test.csv**) sets.


## What we do

1. 
    * Train ensemble models Random Forest (RF) and AdaBoost.M1 with decision stumps as base classifiers. Experiment with different values of hyper-parameters such as number of base classifiers etc.

    * Construct an ensemble classifier using unweighted majority vote over the 4 models you have trained. Report the performance on the test data.

2. 
    * Train 4 individual models: Neural Network (NN), Logistic Regression (LR), Naive Bayes (NB), Decision Tree (DT). Report the confusion matrix and classification accuracy on the test data for each of them. When training the 4 models, slightly tune the hyper-parameters (extensive grid search is not required). Report the experiments we have done.

    * Construct an ensemble classifier using unweighted majority vote over the 4 models we have trained. Report the performance on the test data.

    * Construct an ensemble classifier using weighted majority vote over the 4 models we have trained. Report the performance on the test data. We might use one of the following strategies to decide weights: make weights proportional to the classification accuracy, tune weights as hyperparameters, use stacking, or some other strategies. Report the experiments we have done.



# Specification

Language used: 
* Python 3.8

Additional packages used: 

* numpy 1.18.2

* pandas 1.0.3

* scikit-learn 0.22.2

* scipy 1.4.1	(comes with scikit-learn 0.22.2)

* matplotlib 3.2.1 (optional)

# How to run in command line

* Download **lab4-train.csv** and **lab4-test.csv** files and put them into the working folder.

* Navigate Mac terminal (or Window prompt) to the directory containing the .py file and enter:

    * Fort task 1:
        ```
        $ python RF_ADA.py
        ```
    * For task 2:
        ```
        $ python majority_voting.py
        ```

* Some optionals:

    * If you want to gridsearch hyper-parameters of Random Forest: un-comment lines 35 -> 40 in *RF_ADA.py*

    * If you want to gridsearch hyper-parameters of Ada Boost: un-comment lines 62 -> 66 in *RF_ADA.py*

    * If you want to draw confusion matrix in picture (requires matplotlib):
        
        * of Neural Network	: un-comment lines 57  -> 66 	in majority_voting.py
		* of Logistic Regression: un-comment lines 95  -> 104 	in majority_voting.py
		* of Naive Bayes	: un-comment lines 127 -> 136 	in majority_voting.py
		* of Decision Tree	: un-comment lines 169 -> 178	in majority_voting.py

