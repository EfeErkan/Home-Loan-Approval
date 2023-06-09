import pandas as pd
import numpy as np
import time

'''
    Logistic Function
    @param beta: np.ndarray containing the beta values
    @param x: np.ndarray containing the x values
    @return: float value of the logistic function
'''
def Logistic_Function(beta, x):
    return np.exp(float(x @ beta)) / (1 + np.exp(float(x @ beta)))

'''
    Logistic Regression Training for beta values
    @param df: pandas DataFrame containing data
    @return: np.ndarray containing trained beta values
'''
def Logistic_Regression_Train(df: pd.DataFrame) -> np.ndarray:
    X = df.iloc[:, 0:12].values # Design Matrix
    for i in range(len(X)):
        X[i, 0] = 1
    X = X.astype(float)
    
    y = df['Loan_Status'].values # Response Vector
    y = y.reshape((len(y), 1))
    
    beta_old = np.zeros((12, 1))
    prediction_matrix = np.zeros((len(X), 1))
    weight_matrix = np.zeros((len(X), len(X)))
    
    while True:
        for i in range(len(X)):
            prediction_matrix[i] = Logistic_Function(beta_old, X[i])
        
        for i in range(len(X)):
            weight_matrix[i, i] = prediction_matrix[i] * (1 - prediction_matrix[i])
            
        adjusted_response = X @ beta_old + np.linalg.inv(weight_matrix) @ (y - prediction_matrix)
        
        beta_new = np.linalg.inv(X.T @ weight_matrix @ X) @ X.T @ weight_matrix @ adjusted_response
        
        if np.isclose(beta_old, beta_new).all():
            break
        
        beta_old = beta_new
    
    return beta_new

'''
    Logistic Regression Classifier using trained beta values
    @param beta: np.ndarray containing trained beta values
    @param test_data: dictionary containing test data
    @param threshold: float value of the threshold
    @return: string value of the class label
'''
def Logistic_Regression_Classifier(beta: np.ndarray, test_data, threshold: float = 0.5):
    x = [1]
    for key in test_data.keys():
        x.append(test_data[key])
        
    logistic_val = Logistic_Function(beta, x)
    if logistic_val > threshold:
        return "Y", logistic_val
    else:
        return "N", 1 - logistic_val

'''
    Logistic Regression Test Function
    @param train_df: pandas DataFrame containing data
    @param test_df: pandas DataFrame containing data
    @return: dictionary containing the accuracy, F1 score, and log loss of the Naive Bayes classifier
'''
def Logistic_Regression_Calculate_Measures(train_df: pd.DataFrame, test_df: pd.DataFrame):
    confusion_matrix = np.zeros((2, 2))
    log_loss = 0.0
    
    beta_trained = Logistic_Regression_Train(train_df)
    
    for index, row in test_df.iterrows():
        test_data = {'Gender': row['Gender'], 'Married': row['Married'], 'Dependents': row['Dependents'], 'Education': row['Education'], 'Self_Employed': row['Self_Employed'], 
                     'ApplicantIncome': row['ApplicantIncome'], 'CoapplicantIncome': row['CoapplicantIncome'], 'LoanAmount': row['LoanAmount'], 'Loan_Amount_Term': row['Loan_Amount_Term'], 
                     'Credit_History': row['Credit_History'], 'Property_Area': row['Property_Area']}
        result, p = Logistic_Regression_Classifier(beta_trained, test_data)
        y = 1 if (result == 'Y') else 0
        log_loss += -1 * (y * np.log(p) + (1 - y) * np.log(1 - p))

        if result == 'N' and row['Loan_Status'] == 0: #TN
            confusion_matrix[0,0] += 1
        elif result == 'Y' and row['Loan_Status'] == 0: #FP
            confusion_matrix[0,1] += 1
        elif result == 'N' and row['Loan_Status'] == 1: #FN
            confusion_matrix[1,0] += 1
        elif result == 'Y' and row['Loan_Status'] == 1: #TP
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
    Logistic Regression Average Test Function
    @param df: pandas DataFrame containing data
    @param count: int value of the number of times to run the test
    @return: dictionary containing the average accuracy, F1 score, and log loss of the Naive Bayes classifier
'''
def Logistic_Regression_Test(df: pd.DataFrame, count:int = 10):
    total_accuracy, total_F1_score, total_log_loss, total_time = 0.0, 0.0, 0.0, 0.0

    # Random 20-80 split
    for i in range(count):
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df.sample(frac=0.8, random_state=1)
        test_df = df.drop(train_df.index)
        
        start = time.time()
        result = Logistic_Regression_Calculate_Measures(train_df, test_df)
        end = time.time()
        print(f'Iteration {i + 1} => {result}')
        
        total_accuracy += result['Accuracy']
        total_F1_score += result['F1_Score']
        total_log_loss += result['Log_Loss']
        total_time += end - start
    
    end = time.time()
    
    return {'Average Accuracy': total_accuracy / count, 'Average F1_Score': total_F1_score / count, 'Average Log_Loss': total_log_loss / count, 'Average Time': total_time / count}