import numpy as np
import pandas as pd
import time

CONVERGENCE_BOUNDARY = 1000

'''
    Neural Network with 2 hidden layers
    Hidden layers' activation function: ReLU
    Output layer's activation function: Sigmoid
'''
class Neural_Network:
    '''
        Constructor for Neural_Network class
        @param df: pandas DataFrame containing data
        @param num_of_features: int value of the number of features
        @param n: int value of the number of neurons in the hidden layers
        @param learning_rate: float value of the learning rate
        @param threshold: float value of the threshold
    '''
    def __init__(self, df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, threshold:float):
        self.W1 = np.random.randn(num_of_features + 1, n) - 0.5
        self.W2 = np.random.randn(n, n) - 0.5
        self.W3 = np.random.randn(n, 1) - 0.5
        
        self.X = df.iloc[:, :num_of_features + 1].values
        for i in range(len(self.X)):
            self.X[i, 0] = 1
        self.X = self.X.astype(float)
        self.Y = df['Loan_Status'].values
        self.Y = self.Y.reshape((len(self.Y), 1))
        
        self.learning_rate = learning_rate
        self.threshold = threshold
    
    '''
        RSS Loss Function
        @param A2: np.ndarray containing the output layer's activation values
        @param Y: np.ndarray containing the response vector
        @return: np.ndarray containing the loss values
    '''
    def loss(self, A2, Y):
        return np.square(A2 - Y)
    
    '''
        Forward Propagation
    '''
    def forward_propagation(self):
        Z1 = np.dot(self.X, self.W1)
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(A1, self.W2)
        A2 = Neural_Network.RELU(Z2)
        
        Z3 = np.dot(A2, self.W3)
        A3 = Neural_Network.Sigmoid_Function(Z3)
        
        return Z1, A1, Z2, A2, Z3, A3
    
    '''
        Back Propagation
        @param Z1: np.ndarray containing the hidden layer's linear values
        @param A1: np.ndarray containing the hidden layer's activation values
        @param Z2: np.ndarray containing the hidden layer's linear values
        @param A2: np.ndarray containing the hidden layer's activation values
        @param Z3: np.ndarray containing the output layer's linear values
        @param A3: np.ndarray containing the output layer's activation values
        @return: np.ndarray containing the gradient values
    '''
    def back_propagation(self, Z1, A1, Z2, A2, Z3, A3):
        dZ3 = 2 * (A3 - self.Y) * Neural_Network.Sigmoid_Function_Derivative(Z3)
        dW3 = np.dot(A2.T, dZ3)
        
        dZ2 = np.dot(dZ3, self.W3.T) * Neural_Network.RELU_Derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        
        dZ1 = np.dot(dZ2, self.W2.T) * Neural_Network.RELU_Derivative(Z1)
        dW1 = np.dot(self.X.T, dZ1)
        
        return dW1, dW2, dW3
    
    '''
        Gradient Descent
        @param dW1: np.ndarray containing the gradient values of the first layer
        @param dW2: np.ndarray containing the gradient values of the second layer
        @param dW3: np.ndarray containing the gradient values of the third layer
        @return: np.ndarray containing the updated weight values
    '''
    def gradient_descent(self, dW1, dW2, dW3):
        W1 = self.W1 - self.learning_rate * dW1
        W2 = self.W2 - self.learning_rate * dW2
        W3 = self.W3 - self.learning_rate * dW3
        return W1, W2, W3
    
    '''
        Neural Network Training
    '''
    def train(self):
        count = 0
        while True:
            Z1, A1, Z2, A2, Z3, A3 = self.forward_propagation()
            dW1, dW2, dW3 = self.back_propagation(Z1, A1, Z2, A2, Z3, A3)
            W1, W2, W3 = self.gradient_descent(dW1, dW2, dW3)
            
            if (np.isclose(W1, self.W1, atol=1e-03).all() and np.isclose(W2, self.W2, atol=1e-03).all()) or count > CONVERGENCE_BOUNDARY:
                break
            
            count += 1
            self.W1 = W1
            self.W2 = W2
            self.W3 = W3
    
    '''
        Neural Network Classification after Training
        @param X: np.ndarray containing the input vector
        @return: str value of the classification result
    '''
    def classify(self, X):
        Z1 = np.dot(X, self.W1)
        A1 = Neural_Network.RELU(Z1)
        
        Z2 = np.dot(A1, self.W2)
        A2 = Neural_Network.RELU(Z2)
        
        Z3 = np.dot(A2, self.W3)
        A3 = Neural_Network.Sigmoid_Function(Z3)
        
        result = (A3 > self.threshold).astype(int)
        return ('Y' if result == 1 else 'N'), A3.item()
    
    '''
        Rectified Linear Unit (ReLU) Activation Function
        @param x: np.ndarray containing the input vector
        @return: np.ndarray containing the activation values
    '''
    @staticmethod
    def RELU(x):
        return np.maximum(x, 0)
    
    '''
        Sigmoid Activation Function
        @param x: np.ndarray containing the input vector
        @return: np.ndarray containing the activation values
    '''
    @staticmethod
    def Sigmoid_Function(x):
        x = np.array(x, dtype=np.float32)
        return 1 / (1 + np.exp(-x))
    
    '''
        Rectified Linear Unit (ReLU) Activation Function Derivative
        @param x: np.ndarray containing the input vector
        @return: np.ndarray containing the activation values
    '''
    @staticmethod
    def RELU_Derivative(x):
        return (x > 0).astype(int)
    
    ''' 
        Sigmoid Activation Function Derivative
        @param x: np.ndarray containing the input vector
        @return: np.ndarray containing the activation values
    '''
    @staticmethod
    def Sigmoid_Function_Derivative(x):
        return Neural_Network.Sigmoid_Function(x) * (1 - Neural_Network.Sigmoid_Function(x))

'''
    Neural Network Test Function
    @param train_df: pd.DataFrame containing the training data
    @param test_df: pd.DataFrame containing the test data
    @param num_of_features: int value of the number of features
    @param n: int value of the number of hidden nodes
    @param learning_rate: float value of the learning rate
    @param threshold: float value of the threshold
    @return: dictionary containing the average accuracy, F1 score, and log loss of the neural network
'''
def Neural_Network_Calculate_Measures(train_df:pd.DataFrame, test_df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, threshold=0.5):
    NN = Neural_Network(train_df, num_of_features, n, learning_rate, threshold)
    confusion_matrix = np.zeros((2, 2))
    log_loss = 0.0
    NN.train()
    
    for i in range(len(test_df)):
        test_array = test_df.iloc[i, 0:12].values
        test_array[0] = 1
        result, p = NN.classify(test_array)
        actual = test_df.iloc[i, 12]
        y = 1 if (result == 'Y') else 0
        if p != 0 and p != 1:
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

'''
    Neural Network Cross Validation Function
    @param df: pd.DataFrame containing the data
    @param num_of_features: int value of the number of features
    @param n: int value of the number of hidden nodes
    @param learning_rate: float value of the learning rate
    @param threshold: float value of the threshold
    @param k_fold: int value of the number of folds
    @return: dictionary containing the average accuracy, F1 score, and log loss of the neural network
'''
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
        #print(result)
        accuracy += result['Accuracy']
        F1_Score += result['F1_Score']
        Log_Loss += result['Log_Loss']
        if F1_Score == 0:
            f1_ignore_count += 1
        start += k_fold
        
    end_time = time.time()

    return {"Accuracy": accuracy / np.ceil(size / k_fold), "F1_Score": F1_Score / (np.ceil(size / k_fold) - f1_ignore_count), "Log_Loss": Log_Loss / np.ceil(size / k_fold), "Time": end_time - start_time}

'''
    Neural Network Hyperparameter Tuning Function
    @param df: pd.DataFrame containing the data
'''
def Neural_Network_Hyperparameter_Tuning(df:pd.DataFrame):
    learning_rates = [0.1, 0.05, 0.01]
    num_of_hidden_layer_neurons = [5, 10, 15, 20]
    
    measures = np.empty((len(learning_rates), len(num_of_hidden_layer_neurons)), dtype=object)
    
    for i in range(len(learning_rates)):
        for j in range(len(num_of_hidden_layer_neurons)):
            print(f'learning rate: {learning_rates[i]}, neurons: {num_of_hidden_layer_neurons[j]}')
            result = Neural_Network_Cross_Validation(df, 11, num_of_hidden_layer_neurons[j], learning_rates[i], threshold=0.5, k_fold=10)
            measures[i, j] = result
            print(result)
            print()
    print(measures)
    
'''
    Neural Network Average Test Function
    @param df: pd.DataFrame containing the data
    @param num_of_features: int value of the number of features
    @param n: int value of the number of hidden nodes
    @param learning_rate: float value of the learning rate
    @param threshold: float value of the threshold
    @param count: int value of the number of iterations
    @return: dictionary containing the average accuracy, F1 score, and log loss of the neural network
'''
def Neural_Network_Test(df:pd.DataFrame, num_of_features:int, n:int, learning_rate:float, threshold:float = 0.5, count:int = 10):
    total_accuracy, total_F1_Score, total_log_loss, total_time = 0.0, 0.0, 0.0, 0.0

    for i in range(count):
        # Random 20-80 split
        df = df.sample(frac = 1).reset_index(drop = True)
        train_df = df.sample(frac = 0.8, random_state = 1)
        test_df = df.drop(train_df.index)
        
        start = time.time()
        result = Neural_Network_Calculate_Measures(train_df, test_df, num_of_features, n, learning_rate, threshold)
        end = time.time()
        print(f'Iteration {i + 1} => {result}')
        
        total_accuracy += result['Accuracy']
        total_F1_Score += result['F1_Score']
        total_log_loss += result['Log_Loss']
        total_time += end - start
    
    return {'Average Accuracy': total_accuracy / count, 'Average F1_Score': total_F1_Score / count, 'Average Log_Loss': total_log_loss / count, 'Average Time': total_time / count}