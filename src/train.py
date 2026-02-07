import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
import time
import os
import logging
from pathlib import Path
import numpy as np

# Local imports
from utils import (
    load_data,
    clean_data,
    prepare_data,
    split_data,
    preprocess_targets,
    scale_features,
    get_data_paths,
    save_feature_list,
    SEED,
)
from model import DNNRegressor


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a dataloader.
    Returns avg_loss, mae, r2.
    """
    model.eval()
    total_loss = 0
    all_preds_log, all_targets_log = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_log = model(X_batch)
            loss = criterion(y_pred_log, y_batch)
            total_loss += loss.item() * len(X_batch)
            all_preds_log.append(y_pred_log.cpu())
            all_targets_log.append(y_batch.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds_log = torch.cat(all_preds_log)
    all_targets_log = torch.cat(all_targets_log)

    # Inverse log transform for metrics
    preds_original = torch.expm1(all_preds_log).numpy()
    targets_original = torch.expm1(all_targets_log).numpy()

    # Handle potential NaN values
    if np.isnan(preds_original).any():
        logger = logging.getLogger(__name__)
        logger.warning(
            f"NaN detected in predictions. Replacing with median target value."
        )
        median_val = np.nanmedian(targets_original)
        preds_original = np.nan_to_num(preds_original, nan=median_val)

    mae = mean_absolute_error(targets_original, preds_original)
    r2 = r2_score(targets_original, preds_original)

    return avg_loss, mae, r2


def train(
    data_dir="data", models_dir="models", epochs=100, patience=15, batch_size=128
):
    """
    Main training function.
    """
    print("Starting training pipeline...")

    # Setup directories
    Path(models_dir).mkdir(exist_ok=True)

    # Setup logging
    log_file = os.path.join(models_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Starting training pipeline...")
    logger.info("=" * 50)
    x_path, y_path = get_data_paths(data_dir)
    df = load_data(x_path, y_path)

    logger.info("Cleaning data...")
    df_clean, feature_cols = clean_data(df)

    # Save feature list for inference
    feature_list_path = os.path.join(models_dir, "features.json")
    save_feature_list(feature_cols, feature_list_path)
    logger.info(f"Feature list saved to {feature_list_path}")
    logger.info(f"Number of features: {len(feature_cols)}")

    logger.info("Preparing data...")
    X, y = prepare_data(df_clean, feature_cols)

    # 2. Split data (Train/Val only)
    X_train, X_val, y_train, y_val = split_data(X, y)

    logger.info("Scaling features...")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    X_train_scaled, X_val_scaled, _ = scale_features(
        X_train, X_val, save_path=scaler_path
    )
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")

    # 4. Preprocess targets (log transform)
    y_train_log, y_val_log = preprocess_targets(y_train, y_val)

    # 5. Prepare Tensors and move to device (like notebook)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create tensors AND move to device BEFORE DataLoaders (notebook approach)
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)

    y_train_tensor = torch.FloatTensor(y_train_log).to(device)
    y_val_tensor = torch.FloatTensor(y_val_log).to(device)

    # 6. DataLoaders (data already on device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 7. Model setup (match notebook exactly)
    input_size = X_train.shape[1]
    model = DNNRegressor(input_size, [320, 160, 80, 40], dropout_rate=0.28).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # 8. Training Loop
    logger.info(f"Start training for {epochs} epochs with patience {patience}...")
    logger.info(
        f"Model architecture: input_size={input_size}, hidden_layers=[320, 160, 80, 40]"
    )
    logger.info(f"Optimizer: Adam (lr=0.0007)")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 50)

    best_val_mae = float("inf")
    best_model_path = os.path.join(models_dir, "best_model.pth")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            # Data already on device, no need to transfer

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        avg_train_loss = train_loss / len(train_dataset)

        # Validation
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_mae)  # Use val_mae like notebook
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} MW | Val R2: {val_r2:.4f} | LR: {current_lr:.6f}"
        )

        # Early Stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("  >>> New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(
        f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    logger.info(f"Best Validation MAE: {best_val_mae:.2f} MW")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DNN for Energy Prediction")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Path to data directory"
    )
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()

    train(data_dir=args.data_dir, models_dir=args.models_dir, epochs=args.epochs)
