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

    # Naive Bayes Performance
    print("Naive Bayes Classifier")
    
    print("Cross-Validation with k=1")
    nb_result_1 = Cross_Validation(train_df, Naive_Bayes_Calculate_Measures, k_fold=1)
    print(nb_result_1)
    
    print("Cross-Validation with k=5")
    nb_result_5 = Cross_Validation(train_df, Naive_Bayes_Calculate_Measures, k_fold=5)
    print(nb_result_5)
    
    print("Cross-Validation with k=10")
    nb_result_10 = Cross_Validation(train_df, Naive_Bayes_Calculate_Measures, k_fold=10)
    print(nb_result_10)
    
    # Logistic Regression Performance
    data_reformatting(train_df)
    print("\nLogistic Regression Classifier")
    
    print("Cross-Validation with k=1")
    lr_result_1 = Cross_Validation(train_df, Logistic_Regression_Calculate_Measures, k_fold=1)
    print(lr_result_1)
    
    print("Cross-Validation with k=5")
    lr_result_5 = Cross_Validation(train_df, Logistic_Regression_Calculate_Measures, k_fold=5)
    print(lr_result_5)
    
    print("Cross-Validation with k=10")
    lr_result_10 = Cross_Validation(train_df, Logistic_Regression_Calculate_Measures, k_fold=10)
    print(lr_result_10)
    
    # Plotting the performance
    
    X = ["k=1", "k=5", "k=10"]
    naive_bayes = [nb_result_1["F1_Score"], nb_result_5["F1_Score"], nb_result_10["F1_Score"]]
    logistic_regression = [lr_result_1["F1_Score"], lr_result_5["F1_Score"], lr_result_10["F1_Score"]]
    
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, naive_bayes, 0.4, label = 'Naive Bayes')
    plt.bar(X_axis + 0.2, logistic_regression, 0.4, label = 'Logistic Regression')
    
    plt.xticks(X_axis, X)
    plt.xlabel("K Fold")
    plt.ylabel("F1_Score")
    plt.title("F1_Score Comparison")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()