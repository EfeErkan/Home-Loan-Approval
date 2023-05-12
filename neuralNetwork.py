import numpy as np
import pandas as pd
from preprocessing import *

'''
    Neural Network with 1 hidden layer
    Hidden layer activation function: ReLU
    Output layer activation function: Sigmoid
'''
class Neural_Network:
    def __init__(self, df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float):
        self.W1 = np.random.randn(num_of_features + 1, n) - 0.5
        self.W2 = np.random.randn(n, 1) - 0.5
        
        df = df.sample(frac = 1) #shuffle
        self.X = df.iloc[:, 1:12].values
        self.X = np.append(self.X, np.ones((len(self.X), 1)), axis=1) # Add bias
        self.Y = df.iloc[:, 11:12].values
        
        self.learning_rate = learning_rate
        
    def loss(self, A2, Y):
        return np.square(A2 - Y)
        
    def forward_propagation(self):
        Z1 = np.dot(self.X, self.W1)
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(A1, self.W2)
        A2 = Neural_Network.Sigmoid_Function(Z2)
        
        return Z1, A1, Z2, A2
    
    def back_propagation(self, Z1, A1, Z2, A2):
        dZ2 = 2 * (A2 - self.Y) * Neural_Network.Sigmoid_Function_Derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        
        dZ1 = np.dot(dZ2, self.W2.T) * Neural_Network.RELU_Derivative(Z1)
        dW1 = np.dot(self.X.T, dZ1)
        
        return dW1, dW2
        
    def gradient_descent(self, dW1, dW2):
        W1 = self.W1 - self.learning_rate * dW1
        W2 = self.W2 - self.learning_rate * dW2
        return W1, W2
        
    def train(self):
        count = 0
        while True:
            # Shuffle X
            self.X = self.X[np.random.permutation(self.X.shape[0]), :]
            
            Z1, A1, Z2, A2 = self.forward_propagation()
            dW1, dW2 = self.back_propagation(Z1, A1, Z2, A2)
            W1, W2 = self.gradient_descent(dW1, dW2)
            
            if np.isclose(W1, self.W1, atol=1e-03).all() and np.isclose(W2, self.W2, atol=1e-03).all():
                break
            
            count += 1
            self.W1 = W1
            self.W2 = W2
            
            print(f'loss: {np.mean(self.loss(A2, self.Y))}')
            
        print(f'epochs: {count}')
                
    def classify(self, X):
        Z1 = np.dot(X, self.W1)
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(A1, self.W2)
        A2 = Neural_Network.Sigmoid_Function(Z2)
        
        return A2
    
    @staticmethod
    def RELU(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def Sigmoid_Function(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def RELU_Derivative(x):
        return (x > 0).astype(int)
    
    @staticmethod
    def Sigmoid_Function_Derivative(x):
        return Neural_Network.Sigmoid_Function(x) * (1 - Neural_Network.Sigmoid_Function(x))

# Neural Network Testing
train_df = pd.read_csv('data/loan_sanction_train.csv')
    
non_numeric_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
data_cleaning(train_df, drop_feature_list=non_numeric_features, fill_feature_list=numeric_features)

data_reformatting(train_df, normalize=True)

NN = Neural_Network(train_df, 11, n=8, learning_rate=0.01)
NN.train()