import pandas as pd
import numpy as np

def make_prediction(model, X_test):
    """
    Make predictions using the trained model following the original notebook structure.
    
    Args:
        model: Trained model
        X_test (DataFrame): Test features
        
    Returns:
        array: Predictions
    """
    print("Making predictions...")
    
    try:
        # Make predictions using the trained model
        predictions = model.predict(X_test)
        
        print(f"Predictions made for {len(predictions)} samples")
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None
