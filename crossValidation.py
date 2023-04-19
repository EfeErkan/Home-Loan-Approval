import pandas as pd
import numpy as np
from collections.abc import Callable
import matplotlib.pyplot as plt
from preprocessing import data_reformatting
from naiveBayes import Naive_Bayes_Calculate_Measures
from logisticRegression import Logistic_Regression_Calculate_Measures

def Cross_Validation(df: pd.DataFrame, 
                     measure_calculation_function: Callable[[pd.DataFrame, pd.DataFrame], dict[str, float]], 
                     k_fold: int):
    df = df.sample(frac = 1) #shuffle
    size = len(df)
    accuracy, F1_Score, Log_Loss, start = 0.0, 0.0, 0.0, 0
    f1_ignore_count = 0
    
    while start < size:
        end = start + k_fold
        if end > size:
            end = size
        
        test_df = df.iloc[start : end]
        new_train_df = df.drop(df.index[range(start, end)])
        result = measure_calculation_function(new_train_df, test_df)
        accuracy += result['Accuracy']
        F1_Score += result['F1_Score']
        Log_Loss += result['Log_Loss']
        if F1_Score == 0:
            f1_ignore_count += 1
        start += k_fold

    return {"Accuracy": accuracy / np.ceil(size / k_fold), "F1_Score": F1_Score / (np.ceil(size / k_fold) - f1_ignore_count), "Log_Loss": Log_Loss / np.ceil(size / k_fold)}

def Plot_Cross_Validation_Results(df:pd.DataFrame, measure_type):
    print("Naive Bayes Classifier")
    
    print("Cross-Validation with k=5")
    nb_result_5 = Cross_Validation(df, Naive_Bayes_Calculate_Measures, k_fold=5)
    print(f"{measure_type}: {nb_result_5[measure_type]}")
    
    print("Cross-Validation with k=10")
    nb_result_10 = Cross_Validation(df, Naive_Bayes_Calculate_Measures, k_fold=10)
    print(f"{measure_type}: {nb_result_10[measure_type]}")
    
    # Logistic Regression Performance
    data_reformatting(df)
    print("\nLogistic Regression Classifier")
    
    print("Cross-Validation with k=5")
    lr_result_5 = Cross_Validation(df, Logistic_Regression_Calculate_Measures, k_fold=5)
    print(f"{measure_type}: {lr_result_5[measure_type]}")
    
    print("Cross-Validation with k=10")
    lr_result_10 = Cross_Validation(df, Logistic_Regression_Calculate_Measures, k_fold=10)
    print(f"{measure_type}: {lr_result_10[measure_type]}")
    
    # Plotting the performance
    
    X = ["k=5", "k=10"]
    
    naive_bayes = [nb_result_5[measure_type], nb_result_10[measure_type]]
    logistic_regression = [lr_result_5[measure_type], lr_result_10[measure_type]]
    
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, naive_bayes, 0.4, label = 'Naive Bayes')
    plt.bar(X_axis + 0.2, logistic_regression, 0.4, label = 'Logistic Regression')
    
    plt.xticks(X_axis, X)
    plt.xlabel("K Fold")
    plt.ylabel(measure_type)
    plt.title(f"{measure_type} Comparison")
    plt.legend()
    plt.show()