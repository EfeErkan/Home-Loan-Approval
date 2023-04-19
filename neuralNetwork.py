import numpy as np
import pandas as pd
import random
from preprocessing import data_cleaning, data_reformatting

class Neural_Network:
    def __init__(self, df:pd.DataFrame, num_of_features:int, n:int):
        self.W1 = np.random.randn(n, num_of_features)
        self.W2 = np.random.randn(n, n)
        self.W3 = np.random.randn(1, n)
        
        self.bias1 = np.random.randn(n, 1)
        self.bias2 = np.random.randn(n, 1)
        self.bias3 = np.random.randn()
        
        df = df.sample(frac = 1) #shuffle
        self.X = df.iloc[:, 1:12].values
        self.Y = df.iloc[:, 12].values
        
    def forward_propagation(self):
        random_X = self.X[random.randint(0, len(self.X) - 1), :].reshape(11, 1)
        Z1 = np.dot(self.W1, random_X) + self.bias1
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(self.W2, A1) + self.bias2
        A2 = self.RELU(Z2)
        
        Z3 = np.dot(self.W3, A2) + self.bias3
        A3 = Neural_Network.Sigmoid_Function(Z3)
        
        return (Z1, A1, Z2, A2, Z3, A3)
    
    @staticmethod
    def RELU(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def Sigmoid_Function(x):
        return 1 / (1 + np.exp(-1 * x))

# For testing Dimensionality
train_df = pd.read_csv('data/loan_sanction_train.csv')
non_numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
data_cleaning(train_df, drop_feature_list=non_numeric_features, fill_feature_list=numeric_features)
data_reformatting(train_df)

N = Neural_Network(train_df, 11, 3)
result = N.forward_propagation()
print(result[5].shape)
