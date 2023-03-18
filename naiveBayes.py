import pandas as pd
import math

def naiveBayes(train_df, non_numeric_test_data, numeric_test_data):
    yes_prob = 1
    no_prob = 1
    data_count = float (len(train_df))
    accepted_df = train_df.query("Loan_Status == 'Y'")
    declined_df = train_df.query("Loan_Status == 'N'")
    
    for key in non_numeric_test_data.keys():
        yes_prob *= countFeature(accepted_df, key, non_numeric_test_data[key]) / data_count
        no_prob *= countFeature(declined_df, key, non_numeric_test_data[key]) / data_count

    for key in numeric_test_data.keys():
        yes_mean = accepted_df[key].mean()
        yes_std = accepted_df[key].std()
        yes_prob *= normpdf(numeric_test_data[key], yes_mean, yes_std)

        no_mean = declined_df[key].mean()
        no_std = declined_df[key].std()
        no_prob *= normpdf(numeric_test_data[key], no_mean, no_std)

        print(yes_prob, no_prob)

        if yes_prob > no_prob:
            return "Y"
        else:
            return "N"

def countFeature(df, key, value):
    count = 0
    for row in df.iterrows():
        if row[1][key] == value:
            count += 1

    return count

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom