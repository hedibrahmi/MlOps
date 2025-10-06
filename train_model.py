import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train):
    """
    Train the LightGBM model following the original notebook structure.
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        
    Returns:
        model: Trained LightGBM model
    """
    print("Training LightGBM model...")
    
    # Instantiate the LightGBM regressor model (following notebook)
    lgbm_model = lgb.LGBMRegressor(random_state=42)
    
    # Fit the model to the scaled training data
    lgbm_model.fit(X_train, y_train)
    
    # Make predictions on the training data for evaluation
    y_train_pred_lgbm = lgbm_model.predict(X_train)
    
    # Calculate the Root Mean Squared Error (RMSE) for the LightGBM model
    rmse_lgbm = np.sqrt(mean_squared_error(y_train, y_train_pred_lgbm))
    
    print(f"Root Mean Squared Error on the training data (LightGBM): {rmse_lgbm}")
    print("Model training completed successfully!")
    
    return lgbm_model

