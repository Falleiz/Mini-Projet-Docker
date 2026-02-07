import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
SEED = 123
np.random.seed(SEED)


def get_data_paths(data_dir="data"):
    """
    Returns paths for X and Y data files.
    """
    base_path = Path(data_dir)
    return base_path / "engie_X.csv", base_path / "engie_Y.csv"


def load_data(x_path, y_path):
    """
    Loads and merges X and Y datasets suitable for the project.
    """
    if not x_path.exists():
        raise FileNotFoundError(f"File not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"File not found: {y_path}")

    df_X = pd.read_csv(x_path, sep=";")
    df_Y = pd.read_csv(y_path, sep=";")

    df = df_X.merge(df_Y, on="ID", how="inner")
    return df


def clean_data(df, selected_machine="WT2", training=True, features_to_keep=None):
    """
    Filters for specific machine, removes invalid targets, handles missing values
    and removes outliers based on IQR.

    Args:
        df (pd.DataFrame): Input dataframe.
        selected_machine (str): Machine ID to filter.
        training (bool): If True, performs target filtering and determines cols to drop.
                         If False, keeps rows and enforces features_to_keep.
        features_to_keep (list): List of feature names to strictly enforce during inference.

    Returns:
        pd.DataFrame, list: Cleaned dataframe and list of feature columns.
    """
    # Filter by machine
    if "MAC_CODE" in df.columns:
        df_clean = df[df["MAC_CODE"] == selected_machine].copy()
    else:
        # If MAC_CODE is missing during inference, assume data is already specific to the machine
        df_clean = df.copy()

    if training:
        # Remove TARGET <= 0 only during training
        if "TARGET" in df_clean.columns:
            df_clean = df_clean[df_clean["TARGET"] > 0].copy()

    # Define columns
    metadata_cols = ["ID", "MAC_CODE", "Date_time"]
    target_col = "TARGET"

    if training:
        # Initial feature columns scan
        current_cols = [
            col for col in df_clean.columns if col not in metadata_cols + [target_col]
        ]

        # Drop columns with > 70% missing values
        missing = df_clean[current_cols].isnull().sum()
        missing_pct = (missing / len(df_clean)) * 100
        cols_to_drop = missing[missing_pct > 70].index.tolist()

        feature_cols = [col for col in current_cols if col not in cols_to_drop]
    else:
        # Inference mode: strictly use provided features
        if features_to_keep is None:
            raise ValueError("features_to_keep must be provided for inference mode")
        feature_cols = features_to_keep

        # Ensure all columns exist (add 0-filled if missing)
        for col in feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0

    # Impute missing values
    # Note: ideally we save imputation values from training. For now we run per-batch logic
    # as per notebook pattern, or simplified median fill.
    missing_updated = df_clean[feature_cols].isnull().sum()
    cols_with_missing = missing_updated[missing_updated > 0].index.tolist()

    for col in cols_with_missing:
        valid_data = df_clean[col].dropna()
        if len(valid_data) > 0:
            if training:
                # Shapiro test for normality
                sample_size = min(5000, len(valid_data))
                sample = valid_data.sample(sample_size, random_state=SEED)
                try:
                    _, p_value = shapiro(sample)
                    fill_value = (
                        df_clean[col].mean()
                        if p_value > 0.05
                        else df_clean[col].median()
                    )
                except Exception:
                    fill_value = df_clean[col].median()
            else:
                # Simple median fill for small datasets
                fill_value = df_clean[col].median()

            # Apply fillna (MUST be outside the if/else!)
            df_clean[col] = df_clean[col].fillna(fill_value)
        else:
            # Fallback if column is entirely empty
            df_clean[col] = df_clean[col].fillna(0)

    if training:
        # Outlier detection (Target only as per notebook logic)
        if "TARGET" in df_clean.columns:
            Q1 = df_clean["TARGET"].quantile(0.25)
            Q3 = df_clean["TARGET"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_mask = (df_clean["TARGET"] < lower_bound) | (
                df_clean["TARGET"] > upper_bound
            )
            df_clean = df_clean[~outliers_mask].copy()

    return df_clean, feature_cols


def prepare_data(df, feature_cols):
    """
    Extracts X and y arrays from dataframe.
    Returns X, and y if TARGET exists, else None.
    """
    X = df[feature_cols].values
    if "TARGET" in df.columns:
        y = df["TARGET"].values
        return X, y
    return X, None


def split_data(X, y):
    """
    Splits data EXACTLY like notebook: 70/15/15 (train/val/test).
    We use train and val, ignore test for now.
    """
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )
    # Second split: split temp into 50/50 = 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )
    # Return only train and val (ignore test for deployment training)
    return X_train, X_val, y_train, y_val


def preprocess_targets(y_train, y_val):
    """
    Applies log1p transformation to targets.
    """
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    return y_train_log, y_val_log


def scale_features(X_train, X_val, save_path=None):
    """
    Fits scaler on training data and transforms all sets.
    Optionally saves the scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)

    return X_train_scaled, X_val_scaled, scaler


def save_feature_list(features, path):
    """Saves list of feature names to JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(features, f)


def load_feature_list(path):
    """Loads list of feature names from JSON"""
    with open(path, "r") as f:
        return json.load(f)
