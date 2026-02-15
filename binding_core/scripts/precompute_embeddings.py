    import os
    import sys
    import torch
    import lmdb
    import pickle
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    import pandas as pd

    # Добавляем путь к src
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.utils.dataset import BindingLMDBDataset


    # --- НАСТРОЙКИ ---
    INFERENCE_BATCH_SIZE = 32  
    NUM_WORKERS = 4 
    COMMIT_EVERY = 1000  # ← КЛЮЧЕВАЯ НАСТРОЙКА: коммитим каждые N записей


    class IdAwareDataset(BindingLMDBDataset):
        """
        Обертка, которая возвращает row_id вместе с данными.
        """
        def __getitem__(self, idx):
            data = super().__getitem__(idx)
            # Добавляем ID текущей записи
            data['row_id'] = self.indices[idx]
            return data


    def main():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device: {device}")

        # 1. Пути
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        base_data_dir = Path("../data/baseline") 
        input_db_path = base_data_dir / "db_processed"
        output_db_path = base_data_dir / "db_embeddings"
        index_path = base_data_dir / "index.csv"
        
        models_dir = Path("../models/pretrained")
        prot_path = models_dir / "protbert"
        mol_path = models_dir / "chemberta"

        print(f"📂 Input: {input_db_path}")
        print(f"💾 Output: {output_db_path}")

        # Убеждаемся, что выходная папка существует
        output_db_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Загрузка моделей
        print("🧠 Loading Models...")
        prot_bert = AutoModel.from_pretrained(str(prot_path)).to(device).eval()
        mol_bert = AutoModel.from_pretrained(str(mol_path)).to(device).eval()
        
        # 3. Датасет
        print("📊 Loading Dataset...")
        full_dataset = IdAwareDataset(str(input_db_path), str(index_path))
        total_records = len(full_dataset)
        print(f"   Total records: {total_records}")
        
        loader = DataLoader(
            full_dataset, 
            batch_size=INFERENCE_BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        # 4. LMDB окружение
        # Увеличиваем размер карты для безопасности
        map_size = 100 * 1024 * 1024 * 1024  # 100 GB
        
        print(f"🔐 Opening LMDB: {output_db_path}")
        env = lmdb.open(str(output_db_path), map_size=map_size)
        
        print(f"🔥 Starting Encoding... Batch Size: {INFERENCE_BATCH_SIZE}, Commit Every: {COMMIT_EVERY}")
        
        count = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Encoding", total=len(loader))):
                try:
                    # Перенос на GPU
                    mol_input = {k: v.to(device) for k, v in batch['mol_input'].items()}
                    prot_input = {k: v.to(device) for k, v in batch['prot_input'].items()}
                    
                    # AMP Autocast (FP16)
                    with torch.cuda.amp.autocast():
                        # ProtBERT
                        prot_out = prot_bert(**prot_input)
                        p_mask = prot_input['attention_mask'].unsqueeze(-1).expand(prot_out.last_hidden_state.size()).float()
                        p_emb = torch.sum(prot_out.last_hidden_state * p_mask, 1) / torch.clamp(p_mask.sum(1), min=1e-9)

                        # ChemBERTa
                        mol_out = mol_bert(**mol_input)
                        m_mask = mol_input['attention_mask'].unsqueeze(-1).expand(mol_out.last_hidden_state.size()).float()
                        m_emb = torch.sum(mol_out.last_hidden_state * m_mask, 1) / torch.clamp(m_mask.sum(1), min=1e-9)

                        # Объединение (на CPU)
                        combined = torch.cat((p_emb, m_emb), dim=1).cpu().numpy().astype(np.float32)
                    
                    # Получение целевых значений
                    targets = batch['targets'].numpy().astype(np.float32)
                    row_ids = batch['row_id']  # Тензор с ID из датасета

                    # ✅ КРИТИЧЕСКИ ВАЖНО: Используем txn для каждого батча отдельно
                    txn = env.begin(write=True)
                    try:
                        for i, rid in enumerate(row_ids):
                            save_key = f"emb_{rid}".encode()
                            save_obj = {
                                'embedding': combined[i], 
                                'target': targets[i]
                            }
                            txn.put(save_key, pickle.dumps(save_obj))
                            count += 1
                        
                        # ✅ Коммитим после каждого батча
                        txn.commit()
                        batch_count += 1
                        
                    except Exception as e:
                        print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                        txn.abort()
                        raise e

                    # Выводим прогресс каждые несколько батчей
                    if batch_count % 10 == 0:
                        print(f"   Progress: {count}/{total_records} ({100*count/total_records:.1f}%)")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("\n❌ CUDA Out of Memory!")
                        print(f"   Processed {count} records before OOM.")
                        torch.cuda.empty_cache()
                        sys.exit(1)
                    else:
                        raise e

        # Закрытие LMDB
        env.close()
        print(f"\n✅ SUCCESS! Processed {count} records.")
        print(f"📁 Embeddings saved to: {output_db_path}")


    if __name__ == "__main__":
        main()