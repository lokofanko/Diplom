import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from pathlib import Path

class BindingPredictor(nn.Module):
    def __init__(self, 
                 base_model_dir='models/pretrained',
                 hidden_dim=256, 
                 dropout=0.1,
                 freeze_encoders=True): # <--- ГЛАВНЫЙ СПАСИТЕЛЬ ПАМЯТИ
        super().__init__()
        
        print("🏗️  Инициализация BindingPredictor...")
        
        # Пути к весам
        prot_path = Path(base_model_dir) / 'protbert'
        mol_path = Path(base_model_dir) / 'chemberta'
        
        # 1. Загрузка ProtBERT
        print(f"    Загрузка ProtBERT из {prot_path}...")
        try:
            self.prot_bert = AutoModel.from_pretrained(prot_path)
        except OSError:
            raise OSError(f"Не найдены веса в {prot_path}. Запусти download_models.py!")

        self.prot_hidden = self.prot_bert.config.hidden_size # 1024
        
        # 2. Загрузка ChemBERTa
        print(f"    Загрузка ChemBERTa из {mol_path}...")
        try:
            self.mol_bert = AutoModel.from_pretrained(mol_path)
        except OSError:
            raise OSError(f"Не найдены веса в {mol_path}. Запусти download_models.py!")
            
        self.mol_hidden = self.mol_bert.config.hidden_size # 768
        
        # 3. ЗАМОРОЗКА (FREEZING)
        # Если памяти мало, мы отключаем градиенты у трансформеров.
        # Они работают просто как экстракторы признаков.
        if freeze_encoders:
            print("    ❄️  РЕЖИМ ЗАМОРОЗКИ: Веса трансформеров обновляться не будут (экономия RAM).")
            for param in self.prot_bert.parameters():
                param.requires_grad = False
            for param in self.mol_bert.parameters():
                param.requires_grad = False
        else:
            print("    🔥  РЕЖИМ FINE-TUNING: Осторожно, жрет много памяти!")

        # 4. Fusion Head (Голова)
        # Объединяем вектора (1024 + 768 = 1792) -> превращаем в 1 число
        combined_dim = self.prot_hidden + self.mol_hidden
        
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, prot_input, mol_input):
        # prot_input и mol_input - это словари {'input_ids': ..., 'attention_mask': ...}
        
        # A. Белок
        # Прогоняем через BERT
        prot_out = self.prot_bert(**prot_input)
        
        # Получаем эмбеддинг. Для ProtBERT лучше делать Mean Pooling (успреднение по длине),
        # так как CLS токен иногда работает хуже для задач регрессии.
        # Маска нужна, чтобы не усреднять паддинги (пустые хвосты).
        p_mask = prot_input['attention_mask'].unsqueeze(-1).expand(prot_out.last_hidden_state.size()).float()
        p_emb = torch.sum(prot_out.last_hidden_state * p_mask, 1) / torch.clamp(p_mask.sum(1), min=1e-9)

        # B. Молекула
        mol_out = self.mol_bert(**mol_input)
        # Для ChemBERTa часто используют CLS токен (pooler_output), но Mean Pooling тоже надежен.
        # Возьмем Mean Pooling для консистентности.
        m_mask = mol_input['attention_mask'].unsqueeze(-1).expand(mol_out.last_hidden_state.size()).float()
        m_emb = torch.sum(mol_out.last_hidden_state * m_mask, 1) / torch.clamp(m_mask.sum(1), min=1e-9)
        
        # C. Конкатенация
        combined = torch.cat((p_emb, m_emb), dim=1)
        
        # D. Предсказание
        output = self.head(combined)
        
        return output.squeeze()

if __name__ == "__main__":
    # Тест на работоспособность
    model = BindingPredictor(freeze_encoders=True)
    print("✅ Тест пройден: модель собралась.")