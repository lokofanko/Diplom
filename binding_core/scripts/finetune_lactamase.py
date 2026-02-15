import os
import sys
import yaml
import argparse
import pickle
import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ClearML Check
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("⚠️ ClearML not found. Logging disabled.")

# --- 1. DATASET ---
class LactamaseEmbeddingDataset(Dataset):
    """
    Датасет, читающий эмбеддинги из LMDB по фильтру ID.
    """
    def __init__(self, db_path, index_path, split_ids_path):
        super().__init__()
        self.db_path = db_path
        self.keys = self._map_ids_to_keys(index_path, split_ids_path)
        
        # Открываем LMDB readonly
        self.env = lmdb.open(
            db_path, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )

    def _map_ids_to_keys(self, index_path, split_ids_path):
        """Robust Mapping: Split ID -> Index Row -> LMDB Key"""
        print(f"📖 Mapping IDs from {os.path.basename(split_ids_path)}...")
        
        # Читаем сплит. Пробуем определить хедер.
        try:
            df_split = pd.read_csv(split_ids_path)
            # Если единственная колонка называется row_id, то всё ок
            if 'row_id' in df_split.columns:
                target_ids = df_split['row_id'].astype(str).tolist()
            else:
                # Если хедера нет или он другой -> читаем без хедера
                target_ids = pd.read_csv(split_ids_path, header=None, dtype=str).iloc[:, 0].tolist()
        except:
            target_ids = pd.read_csv(split_ids_path, header=None, dtype=str).iloc[:, 0].tolist()

        # Чистим от заголовков, если они попали в данные
        target_ids = [str(x).strip() for x in target_ids if str(x).lower() not in ['row_id', 'mol_id', 'id']]
        target_set = set(target_ids)
        
        # Читаем индекс
        df = pd.read_csv(index_path, low_memory=False)
        
        # Smart Column Selection
        # Проверяем первый валидный ID
        if not target_ids:
            raise ValueError(f"Split file {split_ids_path} seems empty!")

        sample_id = target_ids[0]
        is_numeric = sample_id.isdigit()
        
        id_col = None
        if is_numeric and 'row_id' in df.columns:
            print(f"   💡 Split IDs look numeric ('{sample_id}') -> Using 'row_id' column.")
            id_col = 'row_id'
        elif 'mol_id' in df.columns:
            print(f"   💡 Using 'mol_id' column (default).")
            id_col = 'mol_id'
        else:
            id_col = df.columns[0]
            print(f"   💡 Fallback to first column '{id_col}'.")
            
        print(f"   ℹ️  Final ID column: '{id_col}'")
        
        df[id_col] = df[id_col].astype(str).str.strip()
        
        # Filter
        mask = df[id_col].isin(target_set)
        df_filtered = df[mask]
        
        valid_keys = []
        for idx in df_filtered.index:
            key = f"emb_{idx}".encode('ascii')
            valid_keys.append(key)
            
        found_count = len(valid_keys)
        if found_count == 0:
            print(f"❌ ERROR: No records found!")
            print(f"   Split Example: '{sample_id}'")
            print(f"   Index Column '{id_col}' Example: '{df[id_col].iloc[0]}'")
            raise ValueError(f"Zero records found for {split_ids_path} using {id_col}")
            
        print(f"✅ Found {found_count} / {len(target_ids)} records.")
        return valid_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        try:
            with self.env.begin(write=False) as txn:
                data = txn.get(key)
            
            if data is None:
                return torch.zeros(1792), torch.tensor(0.0)
                
            obj = pickle.loads(data)
            emb = torch.tensor(obj['embedding'], dtype=torch.float32)
            target = torch.tensor(obj['target'], dtype=torch.float32)
            
            if emb.shape[0] != 1792:
                return torch.zeros(1792), torch.tensor(0.0)
                
            return emb, target
        except:
            return torch.zeros(1792), torch.tensor(0.0)

# --- 2. MODEL ---
class BindingPredictorHead(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- 3. TRAINING LOOP ---
def train_model(config):
    device = torch.device(config['training']['device'])
    print(f"🚀 Device: {device}")
    
    # Paths Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_paths = config['paths']
    
    db_path = os.path.join(base_dir, cfg_paths['embeddings_db'])
    if not os.path.exists(db_path): db_path = cfg_paths['embeddings_db'] 

    index_path = cfg_paths['index_csv']
    if not os.path.isabs(index_path): index_path = os.path.join(base_dir, index_path)
    
    splits_dir = cfg_paths['splits_dir']
    if not os.path.isabs(splits_dir): splits_dir = os.path.join(base_dir, splits_dir)
    
    # 1. Datasets
    print("\n📂 Loading Datasets...")
    train_ds = LactamaseEmbeddingDataset(db_path, index_path, os.path.join(splits_dir, cfg_paths['train_ids']))
    val_ds = LactamaseEmbeddingDataset(db_path, index_path, os.path.join(splits_dir, cfg_paths['val_ids']))
    test_ds = LactamaseEmbeddingDataset(db_path, index_path, os.path.join(splits_dir, cfg_paths['test_ids']))
    
    # num_workers=0 для безопасности с LMDB
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    
    # 2. Model Loading
    print("\n🧠 Loading Base Model...")
    model = BindingPredictorHead(input_dim=1792, hidden_dim=1024).to(device)
    
    base_ckpt = cfg_paths['base_model_path']
    if not os.path.isabs(base_ckpt): base_ckpt = os.path.join(base_dir, base_ckpt)
    
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {base_ckpt}")
        
    state = torch.load(base_ckpt, map_location=device)
    new_state = {}
    for k, v in state.items():
        k = k.replace("head.", "net.").replace("net.", "net.")
        if not k.startswith("net."): k = "net." + k
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    
    # 3. Setup Training
    lr_val = float(config['training']['learning_rate'])
    wd_val = float(config['training']['weight_decay'])
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr_val,
        weight_decay=wd_val
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # ClearML
    if CLEARML_AVAILABLE:
        Task.init(
            project_name=config['clearml']['project_name'],
            task_name=config['clearml']['task_name'],
            output_uri=True
        )

    # 4. Training Loop
    print("\n🔥 Starting Fine-Tuning...")
    best_r2 = -float('inf')
    patience_counter = 0
    save_dir = os.path.join(base_dir, cfg_paths['save_dir'], config['experiment_name'])
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = int(config['training']['epochs'])
    
    for epoch in range(epochs):
        # Train Step
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Step
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        
        print(f"   Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.4f} | Val R²={val_r2:.4f}")
        
        if CLEARML_AVAILABLE:
            Logger.current_logger().report_scalar("Loss", "Train", train_loss, epoch)
            Logger.current_logger().report_scalar("MAE", "Val", val_mae, epoch)
            Logger.current_logger().report_scalar("R2", "Val", val_r2, epoch)
            
        scheduler.step(val_r2)
        
        # Checkpointing
        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_lactamase_model.pt"))
            print("   ⭐ New Best Model Saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= int(config['training']['early_stopping_patience']):
            print("⏹️ Early Stopping.")
            break

    # 5. Final Evaluation (Train, Val, Test)
    print("\n📊 Generating Final Reports (Train/Val/Test)...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_lactamase_model.pt")))
    model.eval()
    
    datasets = {
        "Train": train_loader,
        "Val": val_loader,
        "Test": test_loader
    }
    
    for name, loader in datasets.items():
        print(f"   Processing {name} set...")
        preds, targets = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                p = model(X)
                preds.extend(p.cpu().numpy())
                targets.extend(y.cpu().numpy())
                
        preds = np.array(preds)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)
        
        mask = targets != 0
        mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
        
        print(f"   🏆 {name} Results: MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        # Plot
        plot_path = os.path.join(save_dir, f"results_{name.lower()}.png")
        plot_results(targets, preds, plot_path, mae, r2, name)
        print(f"      Saved plot: {plot_path}")

def plot_results(y_true, y_pred, path, mae, r2, subset_name):
    plt.figure(figsize=(14, 6))
    
    # Scatter
    plt.subplot(1, 2, 1)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(f"{subset_name} Set: Pred vs True\nR² = {r2:.3f}")
    plt.grid(True, alpha=0.3)
    
    # Hist
    plt.subplot(1, 2, 2)
    err = y_true - y_pred
    sns.histplot(err, bins=30, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Error")
    plt.title(f"{subset_name} Error Distribution\nMAE = {mae:.3f}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    
    if CLEARML_AVAILABLE:
        Logger.current_logger().report_image(f"Results {subset_name}", "Plots", local_path=path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune_lactamase.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train_model(config)
