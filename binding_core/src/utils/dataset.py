import lmdb
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class BindingLMDBDataset(Dataset):
    """(Старый класс для сырых данных - оставляем для совместимости, если нужно)"""
    def __init__(self, db_path, indices_path, **kwargs):
        self.db_path = str(db_path)
        self.indices = pd.read_csv(indices_path)['row_id'].values
        self.env = None

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.db_path, readonly=True, lock=False, 
                readahead=False, meminit=False
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._init_env()
        row_id = self.indices[idx]
        key = f'proc_{row_id}'.encode()
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Sample {row_id} not found")
            sample = pickle.loads(data_bytes)
        return {
            'mol_input': {k: torch.from_numpy(v) for k,v in sample['mol_input'].items()},
            'prot_input': {k: torch.from_numpy(v) for k,v in sample['prot_input'].items()},
            'targets': torch.tensor(sample['targets'], dtype=torch.float32)
        }

# --- НОВЫЙ КЛАСС ---
class EmbeddingLMDBDataset(Dataset):
    """
    Датасет для чтения ПРЕКОМПЬЮЧЕННЫХ эмбеддингов.
    Быстрый, легкий, только векторы.
    """
    def __init__(self, db_path, indices_path):
        self.db_path = str(db_path)
        # Читаем CSV. Если там есть заголовок row_id, pandas сам разберется.
        self.indices = pd.read_csv(indices_path)['row_id'].values
        self.env = None

    def _init_env(self):
        if self.env is None:
            # readahead=True здесь полезен, т.к. мы читаем подряд или близко
            self.env = lmdb.open(
                self.db_path, readonly=True, lock=False, 
                readahead=True, meminit=False
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._init_env()
        row_id = self.indices[idx]
        key = f'emb_{row_id}'.encode()
        
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                # На случай если сплиты и база рассинхронены
                raise KeyError(f"Embedding not found for row_id: {row_id}")
            
            # Десериализация (очень быстрая для numpy array)
            sample = pickle.loads(data_bytes)
            
        return {
            'embedding': torch.from_numpy(sample['embedding']), # (1792,)
            'target': torch.tensor(sample['target'], dtype=torch.float32)
        }
