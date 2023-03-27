import pandas as pd
import numpy as np
import math

def Naive_Bayes_Classifier(train_df: pd.DataFrame, non_numeric_test_data, numeric_test_data):
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

def Naive_Bayes_Performance_Evaluater(train_df: pd.DataFrame, k_fold: int):
    df = train_df.sample(frac = 1) #shuffle
    size = len(df)
    accuracy, F1_Score, start = 0.0, 0.0, 0

    while start < size:
        end = start + k_fold
        if end > size:
            end = size
        
        test_df = df.iloc[start : end]
        new_train_df = df.drop(df.index[range(start, end)])
        result = Naive_Bayes_Calculate_Accuracy_and_F1(new_train_df, test_df)
        accuracy += result['Accuracy']
        F1_Score += result['F1_Score']
        start += k_fold

    print(accuracy / np.ceil(size / k_fold), F1_Score / np.ceil(size / k_fold))

def Naive_Bayes_Calculate_Accuracy_and_F1(train_df: pd.DataFrame, test_df: pd.DataFrame):
    confusion_matrix = np.zeros((2, 2))
    
    for index, row in test_df.iterrows():
        non_numeric_test_data = {'Gender': row['Gender'], 'Married': row['Married'], 'Dependents': row['Dependents'], 'Education': row['Education'], 'Self_Employed': row['Self_Employed'], 'Property_Area': row['Property_Area'], 'Credit_History': row['Credit_History']}
        numeric_test_data = {'ApplicantIncome': row['ApplicantIncome'], 'CoapplicantIncome': row['CoapplicantIncome'], 'LoanAmount': row['LoanAmount'], 'Loan_Amount_Term': row['Loan_Amount_Term']}
        result = Naive_Bayes_Classifier(train_df, non_numeric_test_data, numeric_test_data)

        if result == 'N' and row['Loan_Status'] == 'N': #TN
            confusion_matrix[0,0] += 1
        elif result == 'Y' and row['Loan_Status'] == 'N': #FP
            confusion_matrix[0,1] += 1
        elif result == 'N' and row['Loan_Status'] == 'Y': #FN
            confusion_matrix[1,0] += 1
        elif result == 'Y' and row['Loan_Status'] == 'Y': #TP
            confusion_matrix[1,1] += 1
        
    accuracy = float(confusion_matrix[1,1] + confusion_matrix[0,0]) / (confusion_matrix[0,0] + confusion_matrix[0,1] + confusion_matrix[1,0] + confusion_matrix[1,1])
    precision = float(confusion_matrix[1,1]) / (confusion_matrix[1,1] + confusion_matrix[0,1])
    recall = float(confusion_matrix[1,1]) / (confusion_matrix[1,1] + confusion_matrix[1,0])
    F1_Score = 2 * precision * recall / (precision + recall)

    return {'Accuracy': accuracy, 'F1_Score': F1_Score}

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