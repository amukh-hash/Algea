"""
train_nightly_distill.py - Knowledge Distillation (Classification Paradigm)

Trains a lightweight Student model to mimic the Teacher's classification.
Uses KL-Divergence on softened logits + Cross-Entropy on hard labels.

Run AFTER:
1. prep_artifacts.py (generates scaler, HMM, processed data)
2. train_teacher_t5.py (trains and saves Teacher model)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.student import StudentModel
from app.models.teacher import TeacherTransformer

# Optional MLflow
try:
    import mlflow
    MLFLOW_ENABLED = True
except ImportError:
    MLFLOW_ENABLED = False

# --- CONFIGURATION ---
CONFIG = {
    'run_name': 'Bolt_Distill_Nightly_v1',
    'data_path': 'backend/data/processed/train_sequences.npz',
    'teacher_path': 'backend/models/teacher_v1.pt',
    'save_path': 'backend/models/bolt_student_v1.pt',
    
    # Model Params
    'input_dim': 5,       # Features: returns, vol, rsi, spread, regime
    'num_classes': 3,     # 0: Neutral, 1: Buy, 2: Sell
    
    # Distillation Params
    'temperature': 2.0,   # Softens the teacher's logits
    'alpha': 0.5,         # Balance between Hard Label and Soft Teacher
    
    # Training Params
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("DISTILL")


def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    """
    Combines Hard Loss (Truth) and Soft Loss (Teacher Mimicry).
    
    Args:
        student_logits: Unnormalized logits from Student
        teacher_logits: Unnormalized logits from Teacher
        true_labels: Ground truth (0, 1, 2)
        T: Temperature for softening
        alpha: Weight for hard loss (1.0=pure hard, 0.0=pure soft)
    
    Returns:
        total_loss, hard_loss, soft_loss
    """
    # 1. Hard Loss: Student vs Ground Truth
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # 2. Soft Loss: Student vs Teacher
    # KLDiv expects LogSoftmax inputs and Softmax targets
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    
    # KLDivLoss with batchmean reduction
    kl_loss = nn.KLDivLoss(reduction='batchmean')(student_log_soft, teacher_soft)
    
    # Scale by T^2 to keep gradient magnitudes consistent
    soft_loss = kl_loss * (T * T)

    # 3. Weighted Sum
    total_loss = (alpha * hard_loss) + ((1 - alpha) * soft_loss)
    
    return total_loss, hard_loss, soft_loss


def load_data(path, batch_size):
    """Load preprocessed sequence data."""
    logger.info(f"Loading data from {path}...")
    
    if path.endswith('.npz'):
        data = np.load(path)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.int64)
    else:
        raise ValueError(f"Unsupported format: {path}")
    
    tensor_X = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y)
    
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Data Loaded: {len(X)} samples, shape {X.shape}")
    return dataloader, X.shape[2]  # num_features


def run_distillation():
    """Main distillation training loop."""
    
    if MLFLOW_ENABLED:
        mlflow.set_experiment("ALGAI_Distillation")
        mlflow.start_run(run_name=CONFIG['run_name'])
        mlflow.log_params(CONFIG)
    
    logger.info(f"Starting Distillation on {CONFIG['device']}")
    
    # 1. Load Data
    train_loader, num_features = load_data(CONFIG['data_path'], CONFIG['batch_size'])
    
    # 2. Initialize TEACHER (Frozen)
    logger.info("Loading Teacher...")
    
    if not os.path.exists(CONFIG['teacher_path']):
        logger.error(f"Teacher not found at {CONFIG['teacher_path']}")
        logger.error("Run train_teacher_t5.py first!")
        return
    
    checkpoint = torch.load(CONFIG['teacher_path'], map_location=CONFIG['device'])
    
    teacher_model = TeacherTransformer(checkpoint['config']).to(CONFIG['device'])
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()
    
    # Freeze Teacher Weights
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    logger.info(f"Teacher loaded. Config: {checkpoint['config']}")
    
    # 3. Initialize STUDENT (Trainable)
    logger.info("Initializing Student...")
    
    student_config = {
        'input_dim': num_features,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'num_classes': CONFIG['num_classes'],
        'dropout': 0.1
    }
    
    student_model = StudentModel(student_config).to(CONFIG['device'])
    optimizer = optim.AdamW(student_model.parameters(), lr=CONFIG['lr'])
    
    logger.info(f"Student config: {student_config}")
    
    # 4. Training Loop
    logger.info(f"Starting training: {CONFIG['epochs']} epochs")
    
    for epoch in range(CONFIG['epochs']):
        student_model.train()
        total_loss = 0
        total_hard = 0
        total_soft = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(CONFIG['device'])
            y_batch = y_batch.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Teacher Forward (No Grad)
            with torch.no_grad():
                teacher_logits = teacher_model(X_batch)
            
            # Student Forward
            student_logits = student_model(X_batch)
            
            # Loss
            loss, hard_l, soft_l = distillation_loss(
                student_logits,
                teacher_logits,
                y_batch,
                T=CONFIG['temperature'],
                alpha=CONFIG['alpha']
            )
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_hard += hard_l.item()
            total_soft += soft_l.item()
            
            preds = student_logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch Summary
        avg_loss = total_loss / len(train_loader)
        avg_hard = total_hard / len(train_loader)
        avg_soft = total_soft / len(train_loader)
        accuracy = correct / total
        
        logger.info(
            f"Epoch {epoch+1} | Loss: {avg_loss:.4f} "
            f"(Hard: {avg_hard:.4f}, Soft: {avg_soft:.4f}) | Acc: {accuracy:.2%}"
        )
        
        if MLFLOW_ENABLED:
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "hard_loss": avg_hard,
                "soft_loss": avg_soft,
                "train_accuracy": accuracy
            }, step=epoch)

    # 5. Save Student
    logger.info(f"Saving Student to {CONFIG['save_path']}...")
    
    Path(os.path.dirname(CONFIG['save_path'])).mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'config': student_config,
        'input_dim': num_features
    }, CONFIG['save_path'])
    
    if MLFLOW_ENABLED:
        mlflow.log_artifact(CONFIG['save_path'])
        mlflow.end_run()
    
    logger.info("✅ Distillation Complete.")


if __name__ == "__main__":
    run_distillation()
