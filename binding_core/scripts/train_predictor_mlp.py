import sys
import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Task
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Путь к src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.dataset import EmbeddingLMDBDataset

# --- МОДЕЛЬ MLP ---
class BindingPredictorMLP(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1024, dropout=0.2):
        super().__init__()
        
        # Трехслойный перцептрон с BatchNorm и Dropout
        self.net = nn.Sequential(
            # Слой 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Слой 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Слой 3 (Выход)
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Инициализация Kaiming
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x).squeeze()

def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        for batch in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
            inputs = batch['embedding'].to(device)
            targets = batch['target'].to(device)
            
            if is_train:
                optimizer.zero_grad()
                preds = model(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
            else:
                preds = model(inputs)
                loss = criterion(preds, targets)
            
            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    return avg_loss, mae, rmse

def evaluate_and_plot(model, loader, device, name, save_dir, logger):
    """
    Прогоняет датасет через модель, считает метрики и рисует графики.
    """
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {name}"):
            X = batch['embedding'].to(device)
            y = batch['target'].to(device)
            p = model(X)
            preds.extend(p.cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Метрики
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    
    print(f"   🏆 {name} Results: MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%")
    
    # Графики
    plt.figure(figsize=(14, 6))
    
    # Scatter
    plt.subplot(1, 2, 1)
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    sns.scatterplot(x=targets, y=preds, alpha=0.3, s=15)
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(f"{name} Set: Pred vs True\nR² = {r2:.3f}")
    plt.grid(True, alpha=0.3)
    
    # Hist
    plt.subplot(1, 2, 2)
    err = targets - preds
    sns.histplot(err, bins=40, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Error (Actual - Pred)")
    plt.title(f"{name} Error Distribution\nMAE = {mae:.3f}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"results_{name.lower()}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"      Saved plot: {plot_path}")
    
    # ClearML Log
    if logger:
        logger.report_image(f"Results {name}", "Plots", local_path=plot_path)

def main(config_path):
    # 1. Загрузка конфига
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. ClearML Init
    task = Task.init(
        project_name=cfg['project_name'], 
        task_name=cfg['experiment_name']
    )
    task.connect(cfg)
    logger = task.get_logger()

    device = torch.device(cfg['training']['device'])
    print(f"🚀 Device: {device}")

    # 3. Данные
    emb_db_path = cfg['paths']['embeddings_db_path']
    splits_dir = cfg['paths']['splits_dir']
    
    print(f"💿 Loading from: {emb_db_path}")
    
    # Инициализация датасетов
    # Используем имена файлов из конфига (или дефолтные, если в конфиге нет)
    train_file = cfg['paths'].get('train_indices', 'train_indices.csv')
    val_file = cfg['paths'].get('val_indices', 'val_indices.csv')
    test_file = cfg['paths'].get('test_indices', 'test_indices.csv')

    train_ds = EmbeddingLMDBDataset(emb_db_path, os.path.join(splits_dir, train_file))
    val_ds = EmbeddingLMDBDataset(emb_db_path, os.path.join(splits_dir, val_file))
    test_ds = EmbeddingLMDBDataset(emb_db_path, os.path.join(splits_dir, test_file))
    
    kwargs = {
        'batch_size': cfg['training']['batch_size'],
        'num_workers': cfg['training'].get('num_workers', 4),
        'pin_memory': True
    }
    
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    # Для теста можно использовать тот же батч
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    # 4. Модель
    print("🧠 Building MLP...")
    model = BindingPredictorMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        dropout=cfg['model']['dropout']
    ).to(device)

    lr_val = float(cfg['training']['learning_rate'])
    wd_val = float(cfg['training']['weight_decay'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=wd_val)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.MSELoss()

    # 5. Loop
    best_val_mae = float('inf')
    early_stop_counter = 0
    exp_dir = os.path.join(cfg['paths']['save_dir'], cfg['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"🔥 Training for {cfg['training']['epochs']} epochs...")
    
    # Попробуем найти параметр early_stopping или early_stopping_patience
    patience = cfg['training'].get('early_stopping', cfg['training'].get('early_stopping_patience', 5))

    for epoch in range(cfg['training']['epochs']):
        # Train
        t_loss, t_mae, t_rmse = run_epoch(model, train_loader, criterion, optimizer, device, True)
        # Val
        v_loss, v_mae, v_rmse = run_epoch(model, val_loader, criterion, None, device, False)
        
        # Scheduler Step
        scheduler.step(v_loss)

        # Logs
        print(f"Epoch {epoch+1:03d} | Train MAE: {t_mae:.4f} | Val MAE: {v_mae:.4f} | Val RMSE: {v_rmse:.4f}")
        logger.report_scalar("MAE", "Train", iteration=epoch, value=t_mae)
        logger.report_scalar("MAE", "Val", iteration=epoch, value=v_mae)
        logger.report_scalar("Loss", "Val", iteration=epoch, value=v_loss)

        # Save Best
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
            print(f"  💾 New Best Model! (MAE: {best_val_mae:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("🛑 Early Stopping!")
                break
    
    print("🏆 Training Finished. Best Validation MAE:", best_val_mae)

    # 6. Final Evaluation & Plots
    print("\n📊 Generating Final Reports...")
    # Загружаем лучшую модель
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    
    # Рисуем для всех сплитов
    evaluate_and_plot(model, train_loader, device, "Train", exp_dir, logger)
    evaluate_and_plot(model, val_loader, device, "Val", exp_dir, logger)
    evaluate_and_plot(model, test_loader, device, "Test", exp_dir, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/predictor_mlp.yaml")
    args = parser.parse_args()
    main(args.config)
