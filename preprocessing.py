import pandas as pd

# Helper Functions

def convert_column_to_int(df: pd.DataFrame, feature_list):
    for feature in feature_list:
        df[feature] = df[feature].apply(int)

# Fundamental Functions

def data_cleaning(df: pd.DataFrame, drop_feature_list, fill_feature_list) -> None:
    df.dropna(subset=drop_feature_list, inplace=True)
    for feature in fill_feature_list:
        mean_value = df[feature].mean()
        df[feature].fillna(mean_value, inplace=True)
    df['Dependents'].replace('3+', '3', inplace=True)
    convert_column_to_int(df, feature_list=['Credit_History', 'Dependents'])
    