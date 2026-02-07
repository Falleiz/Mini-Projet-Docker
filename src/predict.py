"""
Prediction script for wind power forecasting.
Loads trained model and makes predictions on new data.
"""

import argparse
import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from model import DNNRegressor
from utils import load_data, clean_data, get_data_paths


def predict(data_dir: str, models_dir: str, output_path: str):
    """
    Make predictions using trained model.

    Args:
        data_dir: Directory containing input data
        models_dir: Directory containing trained model and artifacts
        output_path: Path to save predictions CSV
    """
    print("=" * 50)
    print("PREDICTION PIPELINE")
    print("=" * 50)

    # 1. Load model artifacts
    model_path = os.path.join(models_dir, "best_model.pth")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    features_path = os.path.join(models_dir, "features.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}")

    print(f"\n1. Loading model from {model_path}")
    # Load state_dict directly (train.py saves model.state_dict() only)
    state_dict = torch.load(model_path, map_location="cpu")

    print(f"2. Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    print(f"3. Loading feature list from {features_path}")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    print(f"   Features: {len(feature_cols)}")

    # 2. Load and clean data
    print(f"\n4. Loading data from {data_dir}")
    x_path, y_path = get_data_paths(data_dir)
    df = load_data(x_path, y_path)

    print(f"5. Cleaning data (inference mode)")
    # Use inference mode with saved feature list
    df_clean, _ = clean_data(df, training=False, features_to_keep=feature_cols)

    # Prepare features
    X = df_clean[feature_cols].values
    print(f"   Data shape: {X.shape}")

    # 3. Scale features
    print(f"\n6. Scaling features")
    X_scaled = scaler.transform(X)

    # 4. Initialize model
    print(f"\n7. Initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]
    model = DNNRegressor(input_size, [320, 160, 80, 40], dropout_rate=0.28).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"   Device: {device}")

    # 5. Make predictions
    print(f"\n8. Making predictions")
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        predictions_log = model(X_tensor).cpu().numpy()

    # 6. Inverse log transform (CRITICAL!)
    print(f"\n9. Applying inverse transform (expm1)")
    predictions_original = np.expm1(predictions_log)

    print(
        f"   Predictions (log scale) - Min: {predictions_log.min():.4f}, Max: {predictions_log.max():.4f}"
    )
    print(
        f"   Predictions (MW) - Min: {predictions_original.min():.2f}, Max: {predictions_original.max():.2f}"
    )

    # 7. Save predictions
    print(f"\n10. Saving predictions to {output_path}")
    output_df = pd.DataFrame(
        {"ID": df_clean["ID"].values, "PREDICTION_MW": predictions_original}
    )

    output_df.to_csv(output_path, sep=";", index=False)
    print(f"   Saved {len(output_df)} predictions")

    # 8. Summary statistics
    print(f"\n{'='*50}")
    print("PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"Mean prediction: {predictions_original.mean():.2f} MW")
    print(f"Median prediction: {np.median(predictions_original):.2f} MW")
    print(f"Std prediction: {predictions_original.std():.2f} MW")
    print(f"\nPredictions saved to: {output_path}")

    return predictions_original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing input data (default: data)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory containing trained model (default: models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output file for predictions (default: predictions.csv)",
    )

    args = parser.parse_args()

    predict(data_dir=args.data_dir, models_dir=args.models_dir, output_path=args.output)
