#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime

from prepare_data import prepare_data
from train_model import train_model
from evaluate_model import evaluate_model
from save_model import save_model
from load_model import load_model
from predict import make_prediction


def main():
    parser = argparse.ArgumentParser(description='MLOps Pipeline for BPM Prediction')
    parser.add_argument('--mode', choices=['prepare', 'train', 'save', 'load', 'predict', 'evaluate', 'list'], 
                       default='train', help='Mode to run the pipeline')
    parser.add_argument('--train_path', default='train.csv', 
                       help='Path to training data CSV file')
    parser.add_argument('--test_path', default='test.csv', 
                       help='Path to test data CSV file')
    parser.add_argument('--model_name', default='lgbm_model', 
                       help='Name for the model')
    parser.add_argument('--save_dir', default='models', 
                       help='Directory to save models')
    parser.add_argument('--load_version', default='latest', 
                       help='Model version to load (latest or timestamp)')
    parser.add_argument('--output_path', default='predictions.csv', 
                       help='Path to save predictions')
    parser.add_argument('--validation_split', type=float, default=0.2, 
                       help='Validation split ratio for training')
    parser.add_argument('--detailed_eval', action='store_true', 
                       help='Run detailed evaluation with plots')

    args = parser.parse_args()

    print("Webhook test")
    print(f"Starting MLOps Pipeline - Mode: {args.mode}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)

    if args.mode == 'prepare':
        run_prepare_pipeline(args)
    elif args.mode == 'train':
        run_training_pipeline(args)
    elif args.mode == 'save':
        run_save_pipeline(args)
    elif args.mode == 'load':
        run_load_pipeline(args)
    elif args.mode == 'predict':
        run_prediction_pipeline(args)
    elif args.mode == 'evaluate':
        run_evaluation_pipeline(args)
    elif args.mode == 'list':
        list_models(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


def run_prepare_pipeline(args):
    print("Starting Data Preparation Pipeline")
    
    try:
        X_train, y_train, X_test, scaler, features = prepare_data(
            args.train_path, args.test_path
        )
        print("Data preparation completed successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Features: {features}")
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return


def run_training_pipeline(args):
    print("Starting Training Pipeline")
    
    try:
        X_train, y_train, X_test, scaler, features = prepare_data(
            args.train_path, args.test_path
        )
        print("Data preparation completed")
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return
    
    try:
        model = train_model(X_train, y_train)
        print("Model training completed")
    except Exception as e:
        print(f"Error in model training: {e}")
        return
    
    # Note: Evaluation is done separately in evaluate mode
    
    try:
        save_info = save_model(
            model, scaler, args.model_name, args.save_dir
        )
        if save_info is None:
            print("Failed to save model")
            return
        print("Model saved successfully")
        print(f"Model path: {save_info['model_path']}")
        print(f"Scaler path: {save_info['scaler_path']}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    print("Training pipeline completed successfully!")


def run_save_pipeline(args):
    print("Starting Save Pipeline")
    
    try:
        # First load a model to save (use the original model name for loading)
        original_model_name = "lgbm_model"  # Default model name for loading
        model, scaler, metadata = load_model(
            original_model_name, args.save_dir, args.load_version
        )
        if model is None:
            print("Failed to load model for saving")
            return
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        # Save with the specified model name
        save_info = save_model(
            model, scaler, args.model_name, args.save_dir
        )
        if save_info is None:
            print("Failed to save model")
            return
        print("Model saved successfully")
        print(f"Model path: {save_info['model_path']}")
        print(f"Scaler path: {save_info['scaler_path']}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    print("Save pipeline completed successfully!")


def run_load_pipeline(args):
    print("Starting Load Pipeline")
    
    try:
        model, scaler, metadata = load_model(
            args.model_name, args.save_dir, args.load_version
        )
        if model is None:
            print("Failed to load model")
            return
        
        print("Model loaded successfully!")
        print(f"Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"Features: {metadata.get('n_features', 'Unknown')}")
        print(f"Model path: {metadata.get('model_path', 'Unknown')}")
        print(f"Scaler path: {metadata.get('scaler_path', 'Unknown')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Load pipeline completed successfully!")


def run_prediction_pipeline(args):
    print("Starting Prediction Pipeline")
    
    try:
        model, scaler, metadata = load_model(
            args.model_name, args.save_dir, args.load_version
        )
        if model is None:
            print("Failed to load model")
            return
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        import pandas as pd
        test_df = pd.read_csv(args.test_path)
        print(f"Test data loaded: {test_df.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    try:
        # Prepare features (exclude id and target columns)
        numerical_features = test_df.columns.tolist()
        if 'id' in numerical_features:
            numerical_features.remove('id')
        if 'BeatsPerMinute' in numerical_features:
            numerical_features.remove('BeatsPerMinute')
        
        X_test = test_df[numerical_features]
        
        # Scale the test data using the loaded scaler
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features)
        
        print(f"Features prepared: {X_test_scaled.shape}")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return
    
    try:
        predictions = make_prediction(model, X_test_scaled)
        if predictions is None:
            print("Failed to make predictions")
            return
        print(f"Predictions made: {len(predictions)} samples")
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    try:
        import pandas as pd
        predictions_df = pd.DataFrame({
            'id': test_df['id'] if 'id' in test_df.columns else range(len(predictions)),
            'BeatsPerMinute': predictions
        })
        predictions_df.to_csv(args.output_path, index=False)
        print(f"Predictions saved to: {args.output_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return
    
    print("Prediction pipeline completed successfully!")


def run_evaluation_pipeline(args):
    print("Starting Evaluation Pipeline")
    
    try:
        model, scaler, metadata = load_model(
            args.model_name, args.save_dir, args.load_version
        )
        if model is None:
            print("Failed to load model")
            return
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        import pandas as pd
        test_df = pd.read_csv(args.test_path)
        print(f"Test data loaded: {test_df.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    try:
        # Prepare features (exclude id and target columns)
        numerical_features = test_df.columns.tolist()
        if 'id' in numerical_features:
            numerical_features.remove('id')
        if 'BeatsPerMinute' in numerical_features:
            numerical_features.remove('BeatsPerMinute')
        
        X_test = test_df[numerical_features]
        
        # Scale the test data using the loaded scaler
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features)
        
        if 'BeatsPerMinute' in test_df.columns:
            y_test = test_df['BeatsPerMinute']
        else:
            print("No target column found, using dummy targets for evaluation")
            y_test = [0] * len(X_test_scaled)
        
        print(f"Data prepared: X={X_test_scaled.shape}, y={len(y_test)}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    try:
        eval_results = evaluate_model(model, X_test_scaled, y_test, args.model_name)
        print("Model evaluation completed")
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return
    
    print("Evaluation pipeline completed successfully!")


def list_models(args):
    print("Available Models")
    
    try:
        import os
        import json
        
        if not os.path.exists(args.save_dir):
            print("No models directory found")
            return
        
        models = []
        for file in os.listdir(args.save_dir):
            if file.endswith('.json') and 'metadata' in file and not file.endswith('_metadata_latest.json'):
                try:
                    metadata_path = os.path.join(args.save_dir, file)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append(metadata)
                except Exception as e:
                    print(f"Error reading metadata for {file}: {e}")
        
        if not models:
            print("No models found in the specified directory")
            return
        
        print(f"Found {len(models)} model(s):")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['model_name']}")
            print(f"   Timestamp: {model['timestamp']}")
            print(f"   Type: {model['model_type']}")
            print(f"   Features: {model.get('n_features', 'N/A')}")
    except Exception as e:
        print(f"Error listing models: {e}")


if __name__ == "__main__":
    main()

