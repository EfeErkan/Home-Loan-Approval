import pandas as pd
import numpy as np
import math
import time

'''
    Naive Bayes Classifier
    @param train_df: pandas DataFrame containing training data
    @param non_numeric_test_data: dictionary containing non-numeric test data
    @param numeric_test_data: dictionary containing numeric test data
    @return: tuple containing the predicted class and the probability of the prediction
'''
def Naive_Bayes_Classifier(train_df: pd.DataFrame, non_numeric_test_data, numeric_test_data):
    prior_yes_prob = 1
    prior_no_prob = 1
    data_count = len(train_df)
    accepted_df = train_df.query("Loan_Status == 'Y'")
    declined_df = train_df.query("Loan_Status == 'N'")
    
    for key in non_numeric_test_data.keys():
        prior_yes_prob *= countFeature(accepted_df, key, non_numeric_test_data[key]) / len(accepted_df)
        prior_no_prob *= countFeature(declined_df, key, non_numeric_test_data[key]) / len(declined_df)
        
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
        return "Y", prior_yes_prob
    else:
        return "N", prior_no_prob

'''
    Naive Bayes Test Function
    @param train_df: pandas DataFrame containing training data
    @param test_df: pandas DataFrame containing testing data
    @return: dictionary containing the accuracy, F1 score, and log loss of the Naive Bayes classifier
'''
def Naive_Bayes_Calculate_Measures(train_df: pd.DataFrame, test_df: pd.DataFrame):
    confusion_matrix = np.zeros((2, 2))
    log_loss = 0.0
    
    for index, row in test_df.iterrows():
        non_numeric_test_data = {'Gender': row['Gender'], 'Married': row['Married'], 'Dependents': row['Dependents'], 'Education': row['Education'], 'Self_Employed': row['Self_Employed'], 'Property_Area': row['Property_Area'], 'Credit_History': row['Credit_History']}
        numeric_test_data = {'ApplicantIncome': row['ApplicantIncome'], 'CoapplicantIncome': row['CoapplicantIncome'], 'LoanAmount': row['LoanAmount'], 'Loan_Amount_Term': row['Loan_Amount_Term']}
        result, p = Naive_Bayes_Classifier(train_df, non_numeric_test_data, numeric_test_data)
        y = 1 if (result == 'Y') else 0
        log_loss += -1 * (y * np.log(p) + (1 - y) * np.log(1 - p))

        if result == 'N' and row['Loan_Status'] == 'N': #TN
            confusion_matrix[0,0] += 1
        elif result == 'Y' and row['Loan_Status'] == 'N': #FP
            confusion_matrix[0,1] += 1
        elif result == 'N' and row['Loan_Status'] == 'Y': #FN
            confusion_matrix[1,0] += 1
        elif result == 'Y' and row['Loan_Status'] == 'Y': #TP
            confusion_matrix[1,1] += 1
        
    accuracy = (confusion_matrix[1,1] + confusion_matrix[0,0]) / (confusion_matrix[0,0] + confusion_matrix[0,1] + confusion_matrix[1,0] + confusion_matrix[1,1])

    if confusion_matrix[1,1] + confusion_matrix[0,1] == 0 or confusion_matrix[1,1] + confusion_matrix[1,0] == 0:
        F1_Score = 0 # Ignore invalid F1 score
    else:
        precision = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
        recall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
        if precision + recall == 0:
            F1_Score = 0
        else:
            F1_Score = 2 * precision * recall / (precision + recall)

    #print(confusion_matrix)
    return {'Accuracy': accuracy, 'F1_Score': F1_Score, 'Log_Loss': log_loss / len(test_df)}

'''
    Naive Bayes Average Test Function
    @param df: pandas DataFrame containing data
    @param count: number of times to run the test
    @return: dictionary containing the average accuracy, F1 score, and log loss of the Naive Bayes classifier
'''
def Naive_Bayes_Test(df:pd.DataFrame, count:int = 10):
    total_accuracy, total_F1_Score, total_log_loss, total_time = 0.0, 0.0, 0.0, 0.0
    
    # Random 20-80 split
    for i in range(count):
        # shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df.sample(frac=0.8, random_state=1)
        test_df = df.drop(train_df.index)
        
        start_time = time.time()
        result = Naive_Bayes_Calculate_Measures(train_df, test_df)
        end_time = time.time()
        print(f'Iteration {i + 1} => {result}')
        
        total_accuracy += result['Accuracy']
        total_F1_Score += result['F1_Score']
        total_log_loss += result['Log_Loss']
        total_time += end_time - start_time
    
    return {'Average Accuracy': total_accuracy / count, 'Average F1_Score': total_F1_Score / count, 'Average Log_Loss': total_log_loss / count, 'Average Time': total_time / count}

'''
    Count Feature Function
    @param df: pandas DataFrame containing data
    @param key: key to count
    @param value: value to count
    @return: number of times the value appears in the key column
'''
def countFeature(df, key, value):
    count = 0
    for row in df.iterrows():
        if row[1][key] == value:
            count += 1

    return count

'''
    Gaussian Distribution Function for numeric data
    @param x: value to calculate the probability of
    @param mean: mean of the data
    @param std: standard deviation of the data
    @return: probability of x
'''
def gaussian_dist(x, mean, std):
    variance = float(std) ** 2
    denominator = (2 * math.pi * variance) ** 0.5
    numerator = math.exp(-(float(x)-float(mean)) ** 2 / (2 * variance))
    return numerator / denominator