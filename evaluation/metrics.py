import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """
    Return a dictionary of regression metrics.
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, and R2 scores.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
