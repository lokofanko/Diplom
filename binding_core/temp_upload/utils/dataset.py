import lmdb
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset

class BindingLMDBDataset(Dataset):
    def __init__(self, db_path, indices_path, **kwargs):
        # db_path теперь указывает на PROCESSED DB
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
        # Ключ теперь proc_{id}
        key = f'proc_{row_id}'.encode()
        
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Sample {row_id} not found in processed DB")
            sample = pickle.loads(data_bytes)
            
        # Превращаем numpy обратно в тензоры
        return {
            'mol_input': {k: torch.from_numpy(v) for k,v in sample['mol_input'].items()},
            'prot_input': {k: torch.from_numpy(v) for k,v in sample['prot_input'].items()},
            'targets': torch.tensor(sample['targets'], dtype=torch.float32)
        }