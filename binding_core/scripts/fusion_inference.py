import os
import sys
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lmdb
import pickle

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути по умолчанию (подставь свои актуальные)
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "../data/baseline/db_embeddings")
DEFAULT_INDEX_PATH = os.path.join(BASE_DIR, "../data/baseline/index.csv")

# Твой чекпоинт
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "experiments/MLP_Fusion_Emb_Curated_v_ideal_daniil_ubivat/best_model.pt")

# Папка со сплитами
DEFAULT_SPLITS_DIR = "/scratch/ivanb/projects/Diplom/data/baseline/curated_splits/expert_finetune/bcl2_family"

BATCH_SIZE = 1024 # Эмбеддинги легкие, можно большой батч
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATASET (COPY-PASTE из finetune) ---
class EmbeddingLMDBDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, split_file):
        super().__init__()
        self.db_path = db_path
        self.keys = self._load_keys(split_file)
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)

    def _load_keys(self, split_file):
        print(f"📖 Reading IDs from {split_file}...")
        try:
            df = pd.read_csv(split_file)
            # Пытаемся найти колонку row_id
            if 'row_id' in df.columns:
                ids = df['row_id'].astype(str).tolist()
            elif 'mol_id' in df.columns: # fallback
                ids = df['mol_id'].astype(str).tolist()
            else:
                # Если хедера нет или он странный
                ids = pd.read_csv(split_file, header=None).iloc[:, 0].astype(str).tolist()
        except:
            ids = []
            
        # Фильтруем мусор (заголовки)
        ids = [x for x in ids if x.lower() not in ['row_id', 'mol_id', 'id']]
        
        valid_keys = [f"emb_{x}".encode() for x in ids]
        print(f"✅ Found {len(valid_keys)} IDs.")
        return valid_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
        
        if data is None:
            return torch.zeros(1792), torch.tensor(0.0)
            
        obj = pickle.loads(data)
        emb = torch.tensor(obj['embedding'], dtype=torch.float32)
        target = torch.tensor(obj['target'], dtype=torch.float32)
        return emb, target

# --- MODEL (MLP) ---
class BindingPredictorHead(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0), # Dropout выключен на инференсе
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- FUNCTIONS ---
def load_model(path, device):
    print(f"🔮 Loading model: {path}")
    model = BindingPredictorHead()
    state = torch.load(path, map_location=device)
    
    # Fix keys
    new_state = {}
    for k, v in state.items():
        k = k.replace("head.", "net.").replace("net.", "net.")
        if not k.startswith("net."): k = "net." + k
        new_state[k] = v
        
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    return model

def evaluate(model, loader, device):
    preds, targets = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Infer"):
            X, y = X.to(device), y.to(device)
            p = model(X)
            preds.extend(p.cpu().numpy())
            targets.extend(y.cpu().numpy())
    return np.array(preds), np.array(targets)

def plot_and_save(name, preds, targets, save_dir):
    # Metrics
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    
    print(f"🏆 {name} Results: MAE={mae:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    sns.scatterplot(x=targets, y=preds, alpha=0.5)
    plt.title(f"{name}: Pred vs True\nR2={r2:.3f}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.histplot(targets - preds, bins=30, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"{name}: Error Dist\nMAE={mae:.3f}")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_results.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   Saved plot: {save_path}")
    
    # Save CSV
    df = pd.DataFrame({'target': targets, 'predicted': preds, 'error': targets - preds})
    df.to_csv(os.path.join(save_dir, f"{name}_preds.csv"), index=False)

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH)
    parser.add_argument("--output_dir", default="inference_results_emb")
    args = parser.parse_args()
    
    print(f"🚀 Device: {DEVICE}")
    
    # 1. Load Model
    model = load_model(args.model_path, DEVICE)
    
    # 2. Process Splits
    for split_name in ['train', 'val', 'test']:
        # Try finding the file
        found = False
        for fname in [f"{split_name}_indices.csv", f"{split_name}_ids.csv", f"{split_name}.csv"]:
            fpath = os.path.join(args.splits_dir, fname)
            if os.path.exists(fpath):
                print(f"\n📂 Processing {split_name.upper()} ({fpath})...")
                ds = EmbeddingLMDBDataset(args.db_path, fpath)
                if len(ds) == 0:
                    print("   ⚠️ Dataset empty. Check IDs.")
                    continue
                    
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                preds, targets = evaluate(model, loader, DEVICE)
                plot_and_save(split_name, preds, targets, args.output_dir)
                found = True
                break
        
        if not found:
            print(f"⚠️ Split '{split_name}' not found in {args.splits_dir}")
