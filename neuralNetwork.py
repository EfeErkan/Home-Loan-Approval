import numpy as np
import pandas as pd
from preprocessing import *
import time

'''
    Neural Network with 1 hidden layer
    Hidden layer activation function: ReLU
    Output layer activation function: Sigmoid
'''
class Neural_Network:
    def __init__(self, df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, threshold:float):
        self.W1 = np.random.randn(num_of_features + 1, n) - 0.5
        self.W2 = np.random.randn(n, 1) - 0.5
        
        df = df.sample(frac = 1) #shuffle
        self.X = df.iloc[:, 1:12].values
        self.X = np.append(self.X, np.ones((len(self.X), 1)), axis=1) # Add bias
        self.Y = df.iloc[:, 11:12].values
        
        self.learning_rate = learning_rate
        self.threshold = threshold
        
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
        
        result = (A2 > self.threshold).astype(int)
        return ('Y' if result == 1 else 'N'), A2
    
    @staticmethod
    def RELU(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def Sigmoid_Function(x):
        x = np.array(x, dtype=np.float32)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def RELU_Derivative(x):
        return (x > 0).astype(int)
    
    @staticmethod
    def Sigmoid_Function_Derivative(x):
        return Neural_Network.Sigmoid_Function(x) * (1 - Neural_Network.Sigmoid_Function(x))

def Neural_Network_Calculate_Measures(train_df:pd.DataFrame, test_df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, threshold=0.5):
    NN = Neural_Network(train_df, num_of_features, n, learning_rate, threshold)
    confusion_matrix = np.zeros((2, 2))
    log_loss = 0.0
    NN.train()
    
    for i in range(len(test_df)):
        result, p = NN.classify(test_df.iloc[i, 1:12].values)
        actual = test_df.iloc[i, 11:12].values
        y = 1 if (result == 'Y') else 0
        log_loss += -1 * (y * np.log(p) + (1 - y) * np.log(1 - p))
        
        if result == 'N' and actual == 0: #TN
            confusion_matrix[0,0] += 1
        elif result == 'Y' and actual == 0: #FP
            confusion_matrix[0,1] += 1
        elif result == 'N' and actual == 1: #FN
            confusion_matrix[1,0] += 1
        elif result == 'Y' and actual == 1: #TP
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

def Neural_Network_Cross_Validation(df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, threshold:float, k_fold:int):
    df = df.sample(frac = 1) #shuffle
    size = len(df)
    accuracy, F1_Score, Log_Loss, start = 0.0, 0.0, 0.0, 0
    f1_ignore_count = 0
    
    start_time = time.time()
    
    while start < size:
        end = start + k_fold
        if end > size:
            end = size
        
        test_df = df.iloc[start : end]
        new_train_df = df.drop(df.index[range(start, end)])
        result = Neural_Network_Calculate_Measures(new_train_df, test_df, num_of_features, n, learning_rate, threshold)
        accuracy += result['Accuracy']
        F1_Score += result['F1_Score']
        Log_Loss += result['Log_Loss']
        if F1_Score == 0:
            f1_ignore_count += 1
        start += k_fold
        
    end_time = time.time()

    return {"Accuracy": accuracy / np.ceil(size / k_fold), "F1_Score": F1_Score / (np.ceil(size / k_fold) - f1_ignore_count), "Log_Loss": Log_Loss / np.ceil(size / k_fold), "Time": end_time - start_time}