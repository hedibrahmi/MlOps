import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(train_path, test_path):
    """
    Load and preprocess the data following the original notebook structure.
    
    Args:
        train_path (str): Path to the training CSV file
        test_path (str): Path to the test CSV file
        
    Returns:
        tuple: (X_train_scaled, y_train, X_test_scaled, scaler, features)
    """
    print("Loading and preparing data...")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    
    # Identify numerical features (all features except 'id' and target)
    numerical_features = train_df.columns.tolist()
    numerical_features.remove('id')
    if 'BeatsPerMinute' in numerical_features:
        numerical_features.remove('BeatsPerMinute')
    
    print(f"Numerical features: {numerical_features}")
    
    # Check for missing values
    print("Missing values in train_df:\n", train_df.isnull().sum().sum())
    print("Missing values in test_df:\n", test_df.isnull().sum().sum())
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Prepare training data
    X_train = train_df[numerical_features]
    y_train = train_df['BeatsPerMinute']
    
    # Prepare test data
    X_test = test_df[numerical_features]
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features)
    
    # Align columns to ensure consistency
    train_cols = X_train_scaled.columns
    test_cols = X_test_scaled.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_scaled[c] = 0
    
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train_scaled[c] = 0
    
    X_test_scaled = X_test_scaled[train_cols]  # Ensure the order is the same
    
    print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test_scaled: {X_test_scaled.shape}")
    print("Data preparation completed successfully!")
    
    return X_train_scaled, y_train, X_test_scaled, scaler, numerical_features

