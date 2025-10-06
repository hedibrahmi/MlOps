import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model performance following the original notebook structure.
    
    Args:
        model: Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        model_name (str): Name of the model for display
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model...")
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    print(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    print("Model evaluation completed successfully!")
    
    return metrics