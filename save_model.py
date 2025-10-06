import joblib
import json
import os
from datetime import datetime


def save_model(model, scaler, model_name="lgbm_model", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
    scaler_path = os.path.join(save_dir, f"{model_name}_scaler_{timestamp}.pkl")
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata_{timestamp}.json")
    
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        metadata = {
            "model_name": model_name,
            "timestamp": timestamp,
            "model_type": type(model).__name__,
            "scaler_type": type(scaler).__name__,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "features": getattr(model, 'feature_name_', None),
            "n_features": getattr(model, 'n_features_', None),
            "model_params": getattr(model, 'get_params', lambda: {})()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        latest_model_path = os.path.join(save_dir, f"{model_name}_latest.pkl")
        latest_scaler_path = os.path.join(save_dir, f"{model_name}_scaler_latest.pkl")
        latest_metadata_path = os.path.join(save_dir, f"{model_name}_metadata_latest.json")
        
        for old_path, new_path in [(model_path, latest_model_path), 
                                  (scaler_path, latest_scaler_path),
                                  (metadata_path, latest_metadata_path)]:
            try:
                if os.path.exists(new_path) or os.path.islink(new_path):
                    os.remove(new_path)
                os.symlink(os.path.basename(old_path), new_path)
            except (OSError, PermissionError):
                # Fallback to copying files if symlinks fail (e.g., on Windows)
                import shutil
                shutil.copy2(old_path, new_path)
        
        print("Model saved successfully!")
        return {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metadata_path": metadata_path,
            "latest_model_path": latest_model_path,
            "latest_scaler_path": latest_scaler_path,
            "latest_metadata_path": latest_metadata_path,
            "metadata": metadata
        }
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

