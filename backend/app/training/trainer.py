import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from backend.app.core import config
from backend.app.training.dataset import RankingDataset, ranking_collate_fn
from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.rank_losses import listwise_softmax_loss, pairwise_margin_loss

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset
    train_dataset = RankingDataset(
        features_path="backend/data/artifacts/features/features_scaled.parquet",
        priors_dir="backend/data/artifacts/priors",
        date_range=(config.TRAIN_START_DATE, config.TRAIN_SPLIT_DATE)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=ranking_collate_fn)
    
    # 2. Model
    # Expect RankTransformer in backend/app/models/rank_transformer.py
    # Hyperparams: input_dim = feats + priors
    # feats 5 + priors 4 = 9
    model = RankTransformer(input_dim=9, d_model=64, nhead=2, num_layers=2, pooling="none").to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Inputs: [B, L, F]
            feats = batch["features"].to(device)
            priors = batch["priors"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            
            # Concat
            x = torch.cat([feats, priors], dim=-1)
            
            optimizer.zero_grad()
            
            # Forward
            out = model(x, mask) 
            scores = out["score"].squeeze(-1) # [B, L]

            batch_losses = []
            for i in range(scores.size(0)):
                valid = mask[i]
                if valid.sum() < 2:
                    continue
                batch_losses.append(listwise_softmax_loss(scores[i][valid], targets[i][valid]))

            loss = torch.stack(batch_losses).mean() if batch_losses else torch.tensor(0.0, device=device)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")
        
    # Save
    os.makedirs("backend/data/artifacts/models", exist_ok=True)
    torch.save(model.state_dict(), "backend/data/artifacts/models/ranker_v1.pt")
    print("Model saved.")

if __name__ == "__main__":
    train_model()
