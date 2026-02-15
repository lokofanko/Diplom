import sys
import os
import yaml
import lmdb
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Пути
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.tokenizers import MolProteinTokenizer

def main(config_path):
    print("🚀 ЗАПУСК ПРЕПРОЦЕССИНГА ДАННЫХ")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    raw_db_path = cfg['paths']['db_path']
    processed_db_path = cfg['paths']['processed_db_path']
    index_path = os.path.join(os.path.dirname(raw_db_path), 'index.csv')

    # Создаем папку для новой базы
    os.makedirs(processed_db_path, exist_ok=True)

    print("⏳ Загрузка токенизаторов (это займет пару секунд)...")
    tokenizer = MolProteinTokenizer(
        mol_model_path=cfg['paths']['mol_model_path'],
        prot_model_path=cfg['paths']['prot_model_path'],
        mol_max_len=cfg['data']['mol_max_len'],
        prot_max_len=cfg['data']['prot_max_len']
    )

    print("📖 Чтение индекса...")
    df = pd.read_csv(index_path)
    print(f"   Всего записей: {len(df):,}")

    # Открываем сырую базу на чтение
    env_src = lmdb.open(raw_db_path, readonly=True, lock=False)
    
    # Открываем новую базу на запись (размер с запасом 50ГБ)
    env_dst = lmdb.open(processed_db_path, map_size=int(50e9))

    print("⚙️  Начинаем токенизацию и запись...")
    
    batch_size = 1000
    cache = []
    
    with env_src.begin() as txn_src, env_dst.begin(write=True) as txn_dst:
        # Используем tqdm для прогресса
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            row_id = row['row_id']
            
            # Читаем сырые данные
            raw_bytes = txn_src.get(f'sample_{row_id}'.encode())
            if not raw_bytes: continue
            
            sample = pickle.loads(raw_bytes)
            smiles = sample['smiles']
            # Обработка пустого белка (для HiQBind)
            prot_seq = sample['protein_sequence'] if sample['protein_sequence'] else ""
            
            # Токенизируем (возвращает тензоры)
            mol_out, prot_out = tokenizer.tokenize_single(smiles, prot_seq)
            
            # Конвертируем тензоры в numpy (чтобы меньше весили в pickle)
            # Можно хранить и torch tensor, но pickle numpy часто быстрее
            processed_record = {
                'mol_input': {k: v.numpy() for k,v in mol_out.items()},
                'prot_input': {k: v.numpy() for k,v in prot_out.items()},
                'targets': float(sample['pIC50']),
                'row_id': row_id
            }
            
            # Пишем в новую базу
            txn_dst.put(f'proc_{row_id}'.encode(), pickle.dumps(processed_record))
            
    env_src.close()
    env_dst.close()
    print(f"\n✅ Успешно обработано и сохранено в: {processed_db_path}")

if __name__ == "__main__":
    main("configs/predictor_fusion.yaml")