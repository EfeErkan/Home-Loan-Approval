import pandas as pd

'''
    Convert pandas dataframe column string to int
    @param df: pandas dataframe
    @param feature_list: list of features to be converted
'''
def convert_column_to_int(df: pd.DataFrame, feature_list):
    for feature in feature_list:
        df[feature] = df[feature].apply(int)

'''
    Clean data by dropping rows with missing values and filling missing values with mean
    @param df: pandas dataframe
    @param drop_feature_list: list of features to be dropped
    @param fill_feature_list: list of features to be filled with mean
'''
def data_cleaning(df: pd.DataFrame, drop_feature_list, fill_feature_list):
    df.dropna(subset=drop_feature_list, inplace=True)
    for feature in fill_feature_list:
        mean_value = df[feature].mean()
        df[feature].fillna(mean_value, inplace=True)
    df['Dependents'].replace('3+', '3', inplace=True)
    convert_column_to_int(df, feature_list=['Credit_History', 'Dependents'])

'''
    Reformat data by converting string to int for class labels and normalizing data
    @param df: pandas dataframe
    @param normalize: boolean value to normalize data or not
'''
def data_reformatting(df: pd.DataFrame, normalize: bool = False):
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
    
    # Normalize
    if normalize:
        for feature in df.columns:
            if feature != 'Loan_Status' and feature != 'Loan_ID':
                df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()