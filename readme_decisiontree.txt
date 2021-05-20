Readme file for decision tree python files

Decision tree files that are present:

(1) DT_1_decision_tree.py
(2) DT_2_tests.py
(3) DT_3_cross_val.py
(4) DT_4_boundary.py

First all of these files expect the files generated from the preprocessing stage to be in the same folder. My apologies but you will also have to change the path to the files before using

(1) DT_1_decision_tree.py

This file stores the decision tree object thus it is very important that this file is imported into the rest of the files. In the decision tree object there are functions to fit, predict, and evaluate accuracy of the model

(2) DT_2_tests.py

This file plots a graph with max tree depth on the x-axis and misclassification error on the y-axis for both training and testing data as seen in Figure 7. You can also decide whether or not you want to remove certain features that are based on the cross correlation analysis described in the report. Additionally this file will give accuracy of the model and the confusion matrix giving the values reported in the tables.

(3) DT_3_cross_val.py

This file runs a full on 10 fold cross validation for tuning for max tree depth. Additionally it runs it for all three cost functions. It produces Figure 8 in the report.

(4) DT_4_boundary.py

This file makes the decision tree plots (Figure 11) for both datasets. It's important to have both datasets in your folder for it to work.



Thanks for taking the time to read this! Please let me know any suggestions for coding practices in future assignments to make it easier to check my code and mark.

Brendon