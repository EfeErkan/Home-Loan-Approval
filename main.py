import pandas as pd
from preprocessing import *
from logisticRegression import *
from naiveBayes import *
from neuralNetwork import *

def main():
    train_df = pd.read_csv('data/loan_sanction_train.csv')
    
    non_numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    data_cleaning(train_df, drop_feature_list=non_numeric_features, fill_feature_list=numeric_features)
    
    # Naive Bayes Testing
    print(Naive_Bayes_Test(train_df))
    
    # Logistic Regression Testing
    data_reformatting(train_df, normalize=False)
    print(Logistic_Regression_Test(train_df))
    
    # Neural Network Testing
    data_reformatting(train_df, normalize=True)
    print(Neural_Network_Test(train_df, 11, n=20, learning_rate=0.1))
    
if __name__ == '__main__':
    main()