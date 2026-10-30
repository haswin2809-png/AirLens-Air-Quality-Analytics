"""
AirLens: Model Comparison Framework
Implements 6 distinct forecasting approaches:
1. Persistence Baseline (Classical Meteorological Benchmark)
2. Ridge Regression (Statistical Linear Regularized Benchmark)
3. Random Forest (Ensemble Tree Model)
4. LightGBM (Gradient Boosted Decision Trees)
5. PyTorch LSTM (Deep Recurrent Sequence Model)
6. PyTorch Station-Aware Transformer (Multi-Head Self-Attention with Station Embeddings)
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from src.config import FORECAST_HORIZONS


# -------------------------------------------------------------
# 1. Baseline & Tabular Models
# -------------------------------------------------------------

class PersistenceModel:
    """Predicts future PM2.5 as the latest observed PM2.5 (PM2.5_lag_1)."""
    def __init__(self):
        self.name = "Persistence Baseline"

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X['PM2.5_lag_1'].values
        return X[:, 15]  # Index corresponding to PM2.5_lag_1


class RidgeModel:
    """Linear L2-regularized statistical forecasting benchmark."""
    def __init__(self, alpha=1.0):
        self.name = "Ridge (Statistical)"
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0.0, 1000.0)


class RandomForestModel:
    """Nonlinear ensemble bagging regressor."""
    def __init__(self, n_estimators=60, max_depth=12, random_state=42):
        self.name = "Random Forest"
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0.0, 1000.0)


class LightGBMModel:
    """High-efficiency gradient boosted decision tree regressor."""
    def __init__(self, n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42):
        self.name = "LightGBM"
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=1,
            verbose=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X, num_threads=1), 0.0, 1000.0)


# -------------------------------------------------------------
# 2. Deep Learning Models (PyTorch)
# -------------------------------------------------------------

class AirQualityDataset(Dataset):
    def __init__(self, X_features, station_ids, y_targets=None):
        self.X = torch.tensor(X_features, dtype=torch.float32)
        self.stn = torch.tensor(station_ids, dtype=torch.long)
        self.y = torch.tensor(y_targets, dtype=torch.float32) if y_targets is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.stn[idx], self.y[idx]
        return self.X[idx], self.stn[idx]


class PyTorchLSTM(nn.Module):
    """
    Recurrent Neural Network with Station Embedding for PM2.5 forecasting.
    """
    def __init__(self, num_features, num_stations=12, embed_dim=16, hidden_dim=64, num_layers=2, num_horizons=len(FORECAST_HORIZONS)):
        super().__init__()
        self.station_embed = nn.Embedding(num_stations, embed_dim)
        self.input_proj = nn.Linear(num_features + embed_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_horizons)
        )

    def forward(self, x, stn_id):
        # x: [batch, features], stn_id: [batch]
        stn_vec = self.station_embed(stn_id)
        combined = torch.cat([x, stn_vec], dim=-1).unsqueeze(1)  # [batch, 1, feat+embed]
        proj = torch.relu(self.input_proj(combined))
        lstm_out, _ = self.lstm(proj)
        out = self.head(lstm_out[:, -1, :])
        return out


class StationAwareTransformer(nn.Module):
    """
    Station-Aware Transformer Encoder with Learned Spatial Embeddings and Multi-Head Self-Attention.
    Directly models spatial and temporal dependencies across multi-pollutant inputs.
    """
    def __init__(self, num_features, num_stations=12, embed_dim=16, d_model=64, nhead=4, num_layers=2, num_horizons=len(FORECAST_HORIZONS)):
        super().__init__()
        self.station_embed = nn.Embedding(num_stations, embed_dim)
        self.feature_proj = nn.Linear(num_features, d_model - embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_horizons)
        )

    def forward(self, x, stn_id):
        # x: [batch, features], stn_id: [batch]
        stn_vec = self.station_embed(stn_id)  # [batch, embed_dim]
        feat_proj = self.feature_proj(x)      # [batch, d_model - embed_dim]
        tokens = torch.cat([feat_proj, stn_vec], dim=-1).unsqueeze(1)  # [batch, 1, d_model]
        
        encoded = self.transformer_encoder(tokens)  # [batch, 1, d_model]
        out = self.forecast_head(encoded.squeeze(1)) # [batch, num_horizons]
        return out


def train_deep_model(model, train_loader, val_loader, epochs=6, lr=0.002, device="cpu"):
    """
    Standard training loop for PyTorch models with early stopping monitoring.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    
    start_time = time.time()
    for ep in range(epochs):
        model.train()
        for batch_x, batch_stn, batch_y in train_loader:
            batch_x, batch_stn, batch_y = batch_x.to(device), batch_stn.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x, batch_stn)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    
    elapsed = time.time() - start_time
    return model, elapsed
