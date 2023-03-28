import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import data_cleaning, data_reformatting
from naiveBayes import Naive_Bayes_Calculate_Measures
from logisticRegression import Logistic_Regression_Calculate_Measures
from crossValidation import Cross_Validation

def main():
    train_df = pd.read_csv('data/loan_sanction_train.csv')
    
    non_numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    data_cleaning(train_df, drop_feature_list=non_numeric_features, fill_feature_list=numeric_features)
    
    #print(train_df.info())
    
    """ for feature in non_numeric_features:
        print(f'Unique values of {feature}: {train_df[feature].unique()}') """

    n = 5
    k_folds = np.arange(1, n + 1)
    accuracies = np.zeros(n)
    f1_scores = np.zeros(n)
    
    data_reformatting(train_df)
    
    for i in range(n):
        result = Cross_Validation(train_df, Logistic_Regression_Calculate_Measures, k_fold=k_folds[i])
        accuracies[i] = result['Accuracy']
        f1_scores[i] = result['F1_Score']
        print(f"For k = {i + 1}, Accuracy = {accuracies[i]}, F1 Score = {f1_scores[i]}")
        
    plt.plot(k_folds, accuracies, label="Accuracy")
    plt.plot(k_folds, f1_scores, label="F1 Score")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()