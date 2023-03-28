import pandas as pd
import numpy as np
from collections.abc import Callable

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