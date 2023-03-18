import pandas as pd
import math

def Naive_Bayes_Classifier(train_df: pd.DataFrame, non_numeric_test_data, numeric_test_data) -> str:
    prior_yes_prob = 1
    prior_no_prob = 1
    data_count = float (len(train_df))
    accepted_df = train_df.query("Loan_Status == 'Y'")
    declined_df = train_df.query("Loan_Status == 'N'")
    
    for key in non_numeric_test_data.keys():
        prior_yes_prob *= countFeature(accepted_df, key, non_numeric_test_data[key]) / float(len(accepted_df))
        prior_no_prob *= countFeature(declined_df, key, non_numeric_test_data[key]) / float(len(declined_df))
        
    for key in numeric_test_data.keys():
        prior_yes_mean = accepted_df[key].mean()
        prior_yes_std = accepted_df[key].std()
        prior_yes_prob *= gaussian_dist(numeric_test_data[key], prior_yes_mean, prior_yes_std)

        prior_no_mean = declined_df[key].mean()
        prior_no_std = declined_df[key].std()
        prior_no_prob *= gaussian_dist(numeric_test_data[key], prior_no_mean, prior_no_std)
        
    prior_yes_prob *= len(accepted_df) / data_count
    prior_no_prob *= len(declined_df) / data_count

    if prior_yes_prob > prior_no_prob:
        return "Y"
    else:
        return "N"

def countFeature(df, key, value):
    count = 0
    for row in df.iterrows():
        if row[1][key] == value:
            count += 1

    return count

def gaussian_dist(x, mean, std):
    variance = float(std) ** 2
    denominator = (2 * math.pi * variance) ** 0.5
    numerator = math.exp(-(float(x)-float(mean)) ** 2 / (2 * variance))
    return numerator / denominator