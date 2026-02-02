"""
train_teacher_t5.py - Teacher Model Training (Distillation Source)

Key Features:
1. Focal Loss: Handles class imbalance (Buy/Sell vs Neutral)
2. Label Smoothing: Prevents overconfidence, better gradients for Student
3. Temperature Scaling: Softens logits for knowledge transfer
4. HMM Regime as Feature: Pre-computed regime state fed as input

This Teacher produces calibrated probability distributions (logits) that
the Student (Bolt) learns from via KL-Divergence distillation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Optional MLflow tracking
try:
    import mlflow
    MLFLOW_ENABLED = True
except ImportError:
    MLFLOW_ENABLED = False
    print("MLflow not installed. Runs will not be tracked.")

# --- CONFIGURATION ---
CONFIG = {
    'input_dim': 64,        # Features: price, vol, RSI, regime, etc.
    'seq_len': 60,          # Context window (ticks)
    'd_model': 512,         # Teacher capacity (large)
    'nhead': 8,
    'num_layers': 6,        # Deep for high capacity
    'num_classes': 3,       # [0: Neutral, 1: Buy, 2: Sell]
    'dropout': 0.1,
    'lr': 1e-4,
    'batch_size': 32,
    'epochs': 20,
    'temperature': 2.0,     # Soften logits for distillation
    'label_smoothing': 0.1, # Prevent overconfidence
    'focal_gamma': 2.0,     # Focal loss focusing parameter
}

# --- TEACHER ARCHITECTURE ---
class TeacherTransformer(nn.Module):
    """
    High-capacity Transformer Encoder for classification.
    Encoder-only (like BERT) since we're classifying, not generating.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.feature_embedding = nn.Linear(config['input_dim'], config['d_model'])
        
        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, config['d_model']))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'] // 2, config['num_classes'])
        )

    def forward(self, x, return_hidden=False):
        """
        x: [Batch, Seq_Len, Features]
        Returns logits [Batch, num_classes]
        """
        bs, seq_len, _ = x.shape
        
        # Embed and add positional encoding
        x = self.feature_embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Transformer pass
        hidden = self.transformer_encoder(x)
        
        # Pool: take last timestep's embedding
        last_hidden = hidden[:, -1, :]
        
        logits = self.classifier(last_hidden)
        
        if return_hidden:
            return logits, hidden
        return logits
    
    def get_soft_targets(self, x, temperature):
        """Generate softened probability distribution for distillation."""
        logits = self.forward(x)
        return torch.softmax(logits / temperature, dim=-1)

# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    """
    Addresses class imbalance by down-weighting easy (Neutral) examples
    and focusing on hard-to-classify (Buy/Sell) examples.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()

# --- DATA LOADING ---
def load_training_data(config):
    """
    Load preprocessed data with HMM regime as feature.
    Returns DataLoader.
    """
    from app.api.databento_client import DatabentoClient
    from app.features.signal_processing import triple_barrier_labels
    
    # Get data
    client = DatabentoClient(mock_mode=True)
    df = client.get_historical_range("BTC-USD", "2023-01-01", "2023-03-01", schema='mbp-10')
    
    if 'price' not in df.columns:
        df['price'] = (df['bid_px_00'] + df['ask_px_00']) / 2.0
    
    prices = df['price'].values
    
    # Generate features
    # Returns, Volatility
    returns = np.diff(prices) / prices[:-1]
    vol = np.zeros_like(returns)
    for i in range(20, len(returns)):
        vol[i] = np.std(returns[i-20:i])
    
    # HMM Regime (trained in Phase 1)
    try:
        from hmmlearn import GaussianHMM
        hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
        hmm.fit(np.column_stack([returns[20:], vol[20:]]))
        regime = np.zeros(len(returns))
        regime[20:] = hmm.predict(np.column_stack([returns[20:], vol[20:]]))
    except Exception:
        regime = np.zeros(len(returns))
    
    # Triple Barrier Labels
    vol_series = df['price'].rolling(20).std().fillna(method='bfill')
    tbm = triple_barrier_labels(df['price'], vol_series)
    
    # Build sequences
    seq_len = config['seq_len']
    n_samples = len(returns) - seq_len - 32
    
    if n_samples <= 0:
        # Fallback to mock data
        print("Using mock data for demo...")
        X = torch.randn(500, seq_len, config['input_dim'])
        y = torch.randint(0, 3, (500,))
        return DataLoader(TensorDataset(X, y), batch_size=config['batch_size'], shuffle=True)
    
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        # Feature vector per timestep: [return, vol, regime, ...]
        # Pad to input_dim with zeros
        seq_features = np.zeros((seq_len, config['input_dim']))
        for j in range(seq_len):
            idx = i + j
            if idx < len(returns):
                seq_features[j, 0] = returns[idx]
                seq_features[j, 1] = vol[idx]
                seq_features[j, 2] = regime[idx]
        
        X_list.append(seq_features)
        
        # Label at decision point
        label_idx = i + seq_len
        if label_idx < len(tbm):
            y_list.append(tbm.iloc[label_idx])
        else:
            y_list.append(0)
    
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)
    
    # Map -1 (Sell) to 2 for classification
    y[y == -1] = 2
    
    return DataLoader(TensorDataset(X, y), batch_size=config['batch_size'], shuffle=True)

# --- TRAINING LOOP ---
def train_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Teacher on {device}")
    
    # Model
    model = TeacherTransformer(CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = FocalLoss(gamma=CONFIG['focal_gamma'], label_smoothing=CONFIG['label_smoothing'])
    
    # Data
    dataloader = load_training_data(CONFIG)
    
    # MLflow
    if MLFLOW_ENABLED:
        mlflow.start_run(run_name="Teacher_Focal_v1")
        mlflow.log_params(CONFIG)
    
    print(f"Starting training: {CONFIG['epochs']} epochs")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")
        
        if MLFLOW_ENABLED:
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", accuracy, step=epoch)
    
    # Save for distillation
    save_path = "backend/models/teacher_v1.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'temperature': CONFIG['temperature']
    }, save_path)
    
    print(f"Teacher saved to {save_path}")
    print("Ready for Student distillation (train_nightly_distill.py)")
    
    if MLFLOW_ENABLED:
        mlflow.end_run()

if __name__ == "__main__":
    train_teacher()
