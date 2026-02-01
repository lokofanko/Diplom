import sys
import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Task
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Добавляем путь к src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты
from src.architectures.fusion import BindingPredictor
from src.utils.dataset import BindingLMDBDataset
# Collate нам больше не нужен кастомный, берем стандартный
from torch.utils.data._utils.collate import default_collate

def calculate_metrics(preds, targets):
    """Считает метрики (MAE, RMSE, MAPE)."""
    preds = np.array(preds)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    
    # Защита от деления на ноль для MAPE
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - preds) / (targets + epsilon))) * 100
    
    return mae, rmse, mape

def run_epoch(model, loader, criterion, optimizer, device, is_train, logger, global_step, log_interval):
    """Одна эпоха обучения или валидации."""
    model.train(is_train)
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Контекст градиентов
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    # Progress bar
    pbar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    
    with context:
        for batch in pbar:
            # Перенос данных на GPU
            mol_input = {k: v.to(device) for k, v in batch['mol_input'].items()}
            prot_input = {k: v.to(device) for k, v in batch['prot_input'].items()}
            targets = batch['targets'].to(device)
            
            # Forward
            preds = model(prot_input, mol_input)
            loss = criterion(preds, targets)
            
            # Backward (только для Train)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Логирование внутри эпохи
                if logger and (global_step % log_interval == 0):
                    logger.report_scalar("Batch Loss", "Train", iteration=global_step, value=loss.item())
                global_step += 1
                
                pbar.set_postfix({'loss': loss.item()})
            
            total_loss += loss.item()
            
            # Собираем предсказания для метрик (на CPU)
            all_preds.extend(preds.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())
            
    # Считаем метрики за эпоху
    avg_loss = total_loss / len(loader)
    mae, rmse, mape = calculate_metrics(all_preds, all_targets)
    
    return avg_loss, mae, rmse, mape, global_step

def main(config_path):
    # 1. Загрузка конфига
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. ClearML
    task = Task.init(
        project_name=cfg['project_name'], 
        task_name=cfg['experiment_name']
    )
    task.connect(cfg)
    logger = task.get_logger()

    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 3. Данные (Используем ПРЕДОБРАБОТАННУЮ базу)
    print("💿 Loading Preprocessed Data...")
    proc_db_path = cfg['paths']['processed_db_path']
    splits_dir = cfg['paths']['splits_dir']

    # Проверка существования базы
    if not os.path.exists(proc_db_path):
        print(f"❌ ОШИБКА: База {proc_db_path} не найдена!")
        print("   Сначала запустите scripts/preprocess_data.py")
        return

    # Инициализация датасетов (без токенизатора, т.к. данные уже готовы)
    train_ds = BindingLMDBDataset(proc_db_path, os.path.join(splits_dir, 'train_indices.csv'))
    val_ds = BindingLMDBDataset(proc_db_path, os.path.join(splits_dir, 'val_indices.csv'))
    test_ds = BindingLMDBDataset(proc_db_path, os.path.join(splits_dir, 'test_indices.csv'))

    kwargs = {
        'batch_size': cfg['training']['batch_size'],
        'num_workers': cfg['training']['num_workers'],
        'pin_memory': True,
        'collate_fn': None # Используем дефолтный collate, т.к. данные уже тензоры
    }
    
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    # 4. Модель
    print("🧠 Building Model...")
    
    # --- FIX: Убираем лишние параметры перед инициализацией ---
    model_args = cfg['model'].copy()
    if 'type' in model_args:
        del model_args['type'] # Удаляем ключ 'type', который вызывает ошибку
    
    # Определяем папку с предобученными моделями (на уровень выше конкретной модели)
    # cfg['paths']['prot_model_path'] = "../models/pretrained/protbert" -> нам нужно "../models/pretrained"
    base_model_dir = os.path.dirname(cfg['paths']['prot_model_path'])

    model = BindingPredictor(
        base_model_dir=base_model_dir, 
        **model_args
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    criterion = nn.MSELoss()

    # 5. Обучение
    print("🔥 Starting Training...")
    best_val_mae = float('inf') 
    global_step = 0
    
    # Папка для сохранения эксперимента
    exp_dir = os.path.join(cfg['paths']['save_dir'], cfg['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)

    for epoch in range(cfg['training']['epochs']):
        print(f"\n=== EPOCH {epoch+1}/{cfg['training']['epochs']} ===")
        
        # Train
        t_loss, t_mae, t_rmse, t_mape, global_step = run_epoch(
            model, train_loader, criterion, optimizer, device, True, 
            logger, global_step, cfg['training']['log_step_interval']
        )
        print(f"TRAIN | Loss: {t_loss:.4f} | MAE: {t_mae:.4f} | MAPE: {t_mape:.2f}%")
        
        # Val
        v_loss, v_mae, v_rmse, v_mape, _ = run_epoch(
            model, val_loader, criterion, None, device, False, 
            logger, global_step, cfg['training']['log_step_interval']
        )
        print(f"VAL   | Loss: {v_loss:.4f} | MAE: {v_mae:.4f} | MAPE: {v_mape:.2f}%")

        # ClearML Logs
        logger.report_scalar("Epoch MAE", "Train", iteration=epoch, value=t_mae)
        logger.report_scalar("Epoch MAE", "Val", iteration=epoch, value=v_mae)
        logger.report_scalar("Epoch MAPE", "Val", iteration=epoch, value=v_mape)

        # Save Best Model
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            save_path = os.path.join(exp_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path} (MAE: {best_val_mae:.4f})")

    # 6. Финальный тест
    print("\n🧪 Testing Best Model...")
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    
    test_loss, test_mae, test_rmse, test_mape, _ = run_epoch(
        model, test_loader, criterion, None, device, False, None, 0, 0
    )
    
    print(f"\n🏆 FINAL TEST RESULTS:")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    
    # Финальные метрики в ClearML
    logger.report_single_value("Final_Test_MAE", test_mae)
    logger.report_single_value("Final_Test_MAPE", test_mape)
    logger.report_single_value("Final_Test_RMSE", test_rmse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/predictor_fusion.yaml")
    args = parser.parse_args()
    main(args.config)