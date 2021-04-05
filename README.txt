1. 	Language used: Python 3.8

	Additional packages used: 
	
	numpy 1.18.2
	pandas 1.0.3
	scikit-learn 0.22.2.post1
	scipy 1.4.1		(comes with scikit-learn 0.22.1)
	matplotlib 3.2.1    	(optional)


2.	How to run:

	_ Download lab4-train.csv and lab4-test.csv files and put them into the working folder.
	_ Run RF_ADA.py for task 1 and majority_voting.py for task 2.

3. 	Some optionals:

	_ If you want to gridsearch hyper-parameters of Random Forest	: un-comment lines 35 -> 40 in RF_ADA.py
	_ If you want to gridsearch hyper-parameters of Ada Boost 	: un-comment lines 62 -> 66 in RF_ADA.py
	_ If you want to draw confusion matrix in picture (require matplotlib):
		+ of Neural Network	: un-comment lines 57  -> 66 	in majority_voting.py
		+ of Logistic Regression: un-comment lines 95  -> 104 	in majority_voting.py
		+ of Naive Bayes	: un-comment lines 127 -> 136 	in majority_voting.py
		+ of Decision Tree	: un-comment lines 169 -> 178	in majority_voting.py