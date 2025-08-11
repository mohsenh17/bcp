from linear_regression_model import train_linear_regression
from random_forest_model import train_random_forest


def train_model(model_name, X_train, y_train, X_train_scaled=None):
    if model_name == "Linear_Regression":
        return train_linear_regression(X_train_scaled, y_train)
    elif model_name == "Random_Forest":
        return train_random_forest(X_train, y_train)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
