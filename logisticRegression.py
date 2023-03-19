import pandas as pd
import numpy as np

def Logistic_Function(beta, x):
    return np.exp(float(x @ beta)) / (1 + np.exp(float(x @ beta)))

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

def Logistic_Regression_Classifier(beta: np.ndarray, test_data, threshold: float = 0.5) -> str:
    x = [1]
    for key in test_data.keys():
        x.append(test_data[key])
        
    logistic_val = Logistic_Function(beta, x)
    if logistic_val > threshold:
        return "Y"
    else:
        return "N"