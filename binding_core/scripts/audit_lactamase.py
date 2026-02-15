import os
import lmdb
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# --- 1. АРХИТЕКТУРА (MLP) ---
class BindingPredictorHead(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1024, dropout=0.1):
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

# --- 2. УТИЛИТЫ ---
def get_lmdb_keys(target_ids, index_path):
    print(f"📖 Читаем индекс для маппинга: {index_path}")
    try:
        # Читаем index.csv, где первая колонка (или 'id') - это глобальный ID
        # А номер строки (index) - это то, что использовалось для ключа emb_{idx}
        
        # Предполагаем, что в index.csv есть заголовок. Если нет - добавь header=None
        # Обычно там: id, smiles, sequence...
        df = pd.read_csv(index_path) 
        
        # Если колонка с ID называется 'mol_id' или 'id'
        id_col = 'id' if 'id' in df.columns else df.columns[0]
        
        print(f"   Используем колонку '{id_col}' как ID.")
        
        # Делаем колонку ID строковой для сравнения
        df[id_col] = df[id_col].astype(str)
        
        # Фильтруем те строки, ID которых есть в нашем списке лактомаз
        target_set = set(str(x) for x in target_ids)
        mask = df[id_col].isin(target_set)
        
        found_df = df[mask].copy()
        
        # Генерируем ключи: emb_{row_index}
        # Важно: row_index это индекс в DataFrame, если мы читали его целиком без чанков
        # df.index соответствует номеру строки (0, 1, 2...)
        mapping = []
        for idx in found_df.index:
            glob_id = found_df.at[idx, id_col]
            # ВОТ ОН, КЛЮЧЕВОЙ МОМЕНТ: префикс emb_
            key_bytes = f"emb_{idx}".encode()
            mapping.append((glob_id, key_bytes))
            
        return mapping
        
    except Exception as e:
        print(f"⚠️ Ошибка маппинга индексов: {e}")
        return []

def plot_results(y_true, y_pred, output_path):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.title(f"R² = {r2_score(y_true, y_pred):.3f}")
    
    plt.subplot(1, 2, 2)
    sns.histplot(y_true - y_pred, bins=30, kde=True, color='purple')
    plt.title(f"MAE = {mean_absolute_error(y_true, y_pred):.3f}")
    
    plt.savefig(output_path, dpi=300)
    plt.close()

# --- 3. MAIN ---
def main():
    # Пути (Binding Core context)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = os.path.dirname(BASE_DIR) # /scratch/ivanb/projects/Diplom
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "baseline")
    
    IDS_PATH = os.path.join(DATA_ROOT, "splits_transfer", "lactamase_all_ids.csv")
    INDEX_PATH = os.path.join(DATA_ROOT, "index.csv")
    # Обрати внимание: ищем в db_embeddings, раз скрипт генерации создавал именно его
    DB_PATH = os.path.join(DATA_ROOT, "db_embeddings") 
    MODEL_PATH = os.path.join(BASE_DIR, "experiments", "MLP_Fusion_Emb_v1", "best_model.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1. ID Лактомаз
    print(f"📂 ID лактомаз: {IDS_PATH}")
    lact_ids = pd.read_csv(IDS_PATH, header=None, dtype=str).iloc[:, 0].tolist()
    
    # 2. Маппинг (ID -> emb_N)
    key_mapping = get_lmdb_keys(lact_ids, INDEX_PATH)
    if not key_mapping:
        print("❌ Не найдено совпадений в index.csv. Проверь колонку ID.")
        return
    print(f"✅ Найдено ключей: {len(key_mapping)}")

    # 3. Чтение базы
    print(f"📦 Читаем эмбеддинги: {DB_PATH}")
    X_list = []
    y_list = []
    
    # Структура сохраненного объекта из твоего скрипта:
    # {'embedding': np.array(...), 'target': float}
    
    env = lmdb.open(DB_PATH, readonly=True, lock=False)
    with env.begin() as txn:
        for glob_id, key_bytes in tqdm(key_mapping):
            data = txn.get(key_bytes)
            if data is None:
                continue
            
            obj = pickle.loads(data)
            # Извлекаем из словаря
            X_list.append(obj['embedding'])
            y_list.append(obj['target'])
            
    env.close()

    if not X_list:
        print("❌ Ключи сгенерированы (emb_N), но данных в базе нет. База пустая или индексы смещены?")
        return

    # 4. Инференс
    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    y_true = np.array(y_list)
    
    print(f"🧠 Модель: {MODEL_PATH}")
    model = BindingPredictorHead(input_dim=1792, hidden_dim=1024).to(device).eval()
    state = torch.load(MODEL_PATH, map_location=device)
    
    new_state = {}
    for k, v in state.items():
        k = k.replace("head.", "net.")
        if not k.startswith("net."): k = "net." + k
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)

    print("🔥 Инференс...")
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    # 5. Результаты
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    
    print(f"\n📊 MAE: {mae:.4f} | R²: {r2:.4f}")
    
    df_res = pd.DataFrame({"actual": y_true, "pred": preds, "error": np.abs(y_true - preds)})
    df_res.to_csv("lactamase_predictions.csv", index=False)
    plot_results(y_true, preds, "lactamase_audit.png")

if __name__ == "__main__":
    main()
