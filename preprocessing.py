import pandas as pd

def data_cleaning(df: pd.DataFrame, drop_feature_list, fill_feature_list) -> None:
    df.dropna(subset=drop_feature_list, inplace=True)
    for feature in fill_feature_list:
        mean_value = df[feature].mean()
        df[feature].fillna(mean_value, inplace=True)