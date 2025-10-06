import joblib
import json
import os


def load_model(model_name="lgbm_model", save_dir="models", version="latest"):
    if version == "latest":
        model_path = os.path.join(save_dir, f"{model_name}_latest.pkl")
        scaler_path = os.path.join(save_dir, f"{model_name}_scaler_latest.pkl")
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata_latest.json")
    else:
        model_path = os.path.join(save_dir, f"{model_name}_{version}.pkl")
        scaler_path = os.path.join(save_dir, f"{model_name}_scaler_{version}.pkl")
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata_{version}.json")
    
    for file_path in [model_path, scaler_path, metadata_path]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None, None, None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("load_model() function executed successfully!")
        return model, scaler, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None
