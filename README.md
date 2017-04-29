#### 2016.M3.TQF-ML.Loan-Default-Prediciton
Final project for the 'Topics in Quantitative Finance: Machine Learning for Finance' class at PHBS (2016/2017)

## Loan Default Prediction

### Project Summary and Goal

In this project, I will use data from a three-year-old competition on [Kaggle](https://www.kaggle.com/c/loan-default-prediction). The dataset contains a set of standardized, de-trended and anonymized financial transactions with over two hundred thousand observations and 778 features. For each observation, it was recorded whether a default was triggered. In case of a default, the loss was measured. The dependent variable (loss) represents the percentage of the loan that was not repaid by the debtor and therefore takes values 0 to 100. The rest of the features are labeled f1 to f778, so we don’t know what the particular features represent. The task of the competition was to predict whether a loan will default, as well as the loss incurred if it does.

The goal of my project is just to predict the loan defaults using methods that we covered in the TQF-ML course. I will not use the results and hints posted from other competitors, such as a "set of golden features" which can be used to achieve a very high ROC AUC. This set was posted during the competition and all of the winners used these variables. 

The dataset is very large and therefore it will be very important to reduce the dimensionality of the dataset, because otherwise some algorithms would take very long time to finish.

As mentioned before, the dataset comes from a closed competition and therefore the codes of the three winners are posted on the website along with some other samples of codes in the discussions. In this project, I do not attempt to predict the size of the losses, just to get the best accuracy possible. The values for the variables 'loss' were not released anyway, so I could not compare my results with other competitors anyway. Therefore, I only use the training dataset which contains the variable 'loss', so I can check my accuracy. It is therefore a supervised learing.

### Methods

For the classification I only use logistic regression, because the other methods are still computationally expensive even after the data dimensionaly reduction. I tried to use both feature selection and feature extraction methods and I also tried to find the best parameters for the logistic Regression.

For feature extraction, I used principal conponents analysis and for feature selection I used Random Forrest Classifier. Lastly, I tried to reduce the number of features based on the values of their mutual correlations and then I computed a difference of the highest correlated columns and stored those results as new variables to reduce the number of features even more while still maintaining as much information as possible.

### Results

I was not able to achieve such high ROC AUC scores as the competitors who used a set of "golden features" discussed above and the best result I was able to obtain was a 'ROC AUC' of 0.716 and 'Test Accuracy' of 0.906. This result was achieved by running logistic regression with all 731 features, which were left after data preprocessing. After that, I managed to reduce the computational time greatly via the methods for dimensionality reduction explained above and the resuls were declining only slightly. The 'Test accuracy' scores did not prove to be particularly useful as the changes in missclassifications were so small that they did not affect accuracy up to the third decimal point. I also tried other performance metrics based on confusion matrix, in particular Precision, Recall and F1 score, but again the changes were so small that those metrics were not able to tell the difference, or only by very small changes. 

### Additional comments

The main task here is to find the handful of useful features in the sea of the others. Several other techniques can be tried to find the best set of features and those can be then used to find very god results. What is surprising about the competition and the dataset itself is that a ROC AUC score over .9 which I was by far not able to achieve was achieved using only two features! However, how these two features were find remains a mystery, as the solver did not explained how did he obtained them and most of the other competitors (even the best ones) all used them.

For me the goal here was to learn about some machine learning techniques and to try to apply them in practice. I only used logistic regression, but that is only because of computational reasons. I could easily implement SVM with just a few minor changes in the code, but unfortunately I was not able to do so because of computational reasons. It could be done using a fewer rows and then compare the results to the logistic regression, but because I was using the sklearn's pipeline so it would be only a matter of changing a few code components and I would only be able to obtain a conclusion that SVM is better/worse/same as logistic regression and I would not really learn anything new about machine learning methods, so I will leave this for a future research.
