import pandas as pd
from preprocessing import data_cleaning
from naiveBayes import naiveBayes

def main():
    train_df = pd.read_csv('data/loan_sanction_train.csv')
    
    non_numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    data_cleaning(train_df, drop_feature_list=non_numeric_features, fill_feature_list=numeric_features)
    
    print(train_df.info())
    
    for feature in non_numeric_features:
        print(f'Unique values of {feature}: {train_df[feature].unique()}')

    non_numeric_test_data = {'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0, 'Education': 'Graduate', 'Self_Employed': 'No', 'Property_Area': 'Urban', 'Credit_History': 0}
    numeric_test_data = {'ApplicantIncome': 0, 'CoapplicantIncome': 0, 'LoanAmount': 100, 'Loan_Amount_Term': 600}
    
    result = naiveBayes(train_df, non_numeric_test_data, numeric_test_data)
    print(result)

if __name__ == '__main__':
    main()