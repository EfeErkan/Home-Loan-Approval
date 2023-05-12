import pandas as pd

# Helper Functions

def convert_column_to_int(df: pd.DataFrame, feature_list):
    for feature in feature_list:
        df[feature] = df[feature].apply(int)

# Fundamental Functions

def data_cleaning(df: pd.DataFrame, drop_feature_list, fill_feature_list):
    df.dropna(subset=drop_feature_list, inplace=True)
    for feature in fill_feature_list:
        mean_value = df[feature].mean()
        df[feature].fillna(mean_value, inplace=True)
    df['Dependents'].replace('3+', '3', inplace=True)
    convert_column_to_int(df, feature_list=['Credit_History', 'Dependents'])
    
def data_reformatting(df: pd.DataFrame, regularize: bool = False):
    df['Gender'].replace('Male', 0, inplace=True)
    df['Gender'].replace('Female', 1, inplace=True)
    
    df['Married'].replace('No', 0, inplace=True)
    df['Married'].replace('Yes', 1, inplace=True)
    
    df['Education'].replace('Not Graduate', 0, inplace=True)
    df['Education'].replace('Graduate', 1, inplace=True)
    
    df['Self_Employed'].replace('No', 0, inplace=True)
    df['Self_Employed'].replace('Yes', 1, inplace=True)
    
    df['Property_Area'].replace('Rural', 0, inplace=True)
    df['Property_Area'].replace('Semiurban', 1, inplace=True)
    df['Property_Area'].replace('Urban', 2, inplace=True)
    
    df['Loan_Status'].replace('N', 0, inplace=True)
    df['Loan_Status'].replace('Y', 1, inplace=True)
    
    # Regularization
    if regularize:
        df['ApplicantIncome'] = (df['ApplicantIncome'] - df['ApplicantIncome'].mean()) / df['ApplicantIncome'].std()
        df['CoapplicantIncome'] = (df['CoapplicantIncome'] - df['CoapplicantIncome'].mean()) / df['CoapplicantIncome'].std()
        df['LoanAmount'] = (df['LoanAmount'] - df['LoanAmount'].mean()) / df['LoanAmount'].std()
        df['Loan_Amount_Term'] = (df['Loan_Amount_Term'] - df['Loan_Amount_Term'].mean()) / df['Loan_Amount_Term'].std()