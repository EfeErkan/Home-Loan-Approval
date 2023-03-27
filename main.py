import time
import pandas as pd
from preprocessing import data_cleaning, data_reformatting
from naiveBayes import Naive_Bayes_Classifier, Naive_Bayes_Performance_Evaluater
from logisticRegression import Logistic_Regression_Train, Logistic_Regression_Classifier

def main():
    train_df = pd.read_csv('data/loan_sanction_train.csv')
    
    non_numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    data_cleaning(train_df, drop_feature_list=non_numeric_features, fill_feature_list=numeric_features)
    
    #print(train_df.info())
    
    """ for feature in non_numeric_features:
        print(f'Unique values of {feature}: {train_df[feature].unique()}') """

    # Naive Bayes Classifier
    
    non_numeric_test_data = {'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0, 'Education': 'Graduate', 'Self_Employed': 'No', 'Property_Area': 'Urban', 'Credit_History': 1}
    numeric_test_data = {'ApplicantIncome': 5720, 'CoapplicantIncome': 0, 'LoanAmount': 110, 'Loan_Amount_Term': 360}
    
    timm = time.time() * 1000
    result = Naive_Bayes_Classifier(train_df, non_numeric_test_data, numeric_test_data)
    print("Naive Bayes Classifier", time.time() * 1000 - timm, result)
    
    print(Naive_Bayes_Performance_Evaluater(train_df, k_fold=10))
    # Logistic Regression
    
    test_data = {'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': 1, 'Self_Employed': 0,
                 'ApplicantIncome': 5720, 'CoapplicantIncome': 0, 'LoanAmount': 110,
                 'Loan_Amount_Term': 360, 'Credit_History': 1, 'Property_Area': 2}
    
    data_reformatting(train_df)
    timm = time.time() * 1000
    beta = Logistic_Regression_Train(train_df)
    result = Logistic_Regression_Classifier(beta, test_data)
    print("Logistic Regression Classifier", time.time() * 1000 - timm, result)

if __name__ == '__main__':
    main()