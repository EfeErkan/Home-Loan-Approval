import numpy as np
import pandas as pd

'''
    Neural Network with 1 hidden layer
    Hidden layer activation function: ReLU
    Output layer activation function: Sigmoid
'''
class Neural_Network:
    def __init__(self, df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, epochs:int):
        self.W1 = np.random.randn(num_of_features, n)
        self.W2 = np.random.randn(n, 1)
        
        self.bias1 = np.random.randn(n)
        self.bias2 = np.random.randn(1)
        
        df = df.sample(frac = 1) #shuffle
        self.X = df.iloc[:, 1:12].values
        self.Y = df.iloc[:, 12].values
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def loss(self, A2, Y):
        return np.square(A2 - Y)
        
    def forward_propagation(self):
        Z1 = np.dot(self.X, self.W1) + self.bias1
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(A1, self.W2) + self.bias2
        A2 = Neural_Network.Sigmoid_Function(Z2)
        
        return Z1, A1, Z2, A2
    
    def backward_propagation(self, Z1, A1, Z2, A2):
        dZ2 = 2 * (A2 - self.Y) * Neural_Network.Sigmoid_Function_Derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0)
        
        dZ1 = np.dot(dZ2, self.W2.T) * Neural_Network.RELU_Derivative(Z1)
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0)
        
        return dW1, db1, dW2, db2
        
    def gradient_descent(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 = self.W1 - learning_rate * dW1
        self.W2 = self.W2 - learning_rate * dW2
        self.bias1 = self.bias1 - learning_rate * db1
        self.bias2 = self.bias2 - learning_rate * db2
        
    def train(self):
        for i in range(self.epochs):
            Z1, A1, Z2, A2 = self.forward_propagation()
            dW1, db1, dW2, db2 = self.backward_propagation(Z1, A1, Z2, A2)
            self.gradient_descent(dW1, db1, dW2, db2, self.learning_rate)
            
            if i % 100 == 0:
                print('Loss: ', self.loss(A2, self.Y))
                
    def classify(self, X):
        Z1 = np.dot(X, self.W1) + self.bias1
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(A1, self.W2) + self.bias2
        A2 = Neural_Network.Sigmoid_Function(Z2)
        
        return A2
    
    @staticmethod
    def RELU(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def Sigmoid_Function(x):
        return 1 / (1 + np.exp(-1 * x))
    
    @staticmethod
    def RELU_Derivative(x):
        return 1 * (x > 0)
    
    @staticmethod
    def Sigmoid_Function_Derivative(x):
        return Neural_Network.Sigmoid_Function(x) * (1 - Neural_Network.Sigmoid_Function(x))
