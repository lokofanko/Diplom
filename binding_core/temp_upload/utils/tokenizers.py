import torch
from transformers import AutoTokenizer, BertTokenizer

class MolProteinTokenizer:
    def __init__(self, mol_model_path, prot_model_path, mol_max_len, prot_max_len):
        print(f"Tokenizer init: loading from {prot_model_path}...")
        
        # ChemBERTa
        self.mol_tokenizer = AutoTokenizer.from_pretrained(mol_model_path)
        
        # ProtBERT (Rostlab)
        # Лучше использовать BertTokenizer напрямую с do_lower_case=False
        try:
            self.prot_tokenizer = BertTokenizer.from_pretrained(prot_model_path, do_lower_case=False)
        except Exception:
            print("⚠️ Failed to load as BertTokenizer, falling back to AutoTokenizer")
            self.prot_tokenizer = AutoTokenizer.from_pretrained(prot_model_path)
            
        self.mol_max_len = mol_max_len
        self.prot_max_len = prot_max_len

    def tokenize_batch(self, smiles_list, prot_sequences_list):
        # Добавляем пробелы между аминокислотами, если их нет
        # ProtBERT обучался на данных с пробелами: "M E T ..."
        processed_seqs = []
        for seq in prot_sequences_list:
            if not seq:
                processed_seqs.append("")
            elif " " not in seq:
                processed_seqs.append(" ".join(list(seq)))
            else:
                processed_seqs.append(seq)
        
        mol_encodings = self.mol_tokenizer(
            smiles_list, 
            padding='max_length', 
            truncation=True, 
            max_length=self.mol_max_len, 
            return_tensors='pt'
        )
        
        prot_encodings = self.prot_tokenizer(
            processed_seqs, 
            padding='max_length', 
            truncation=True, 
            max_length=self.prot_max_len, 
            return_tensors='pt'
        )
        
        return mol_encodings, prot_encodings

    def tokenize_single(self, smiles, prot_sequence):
        mol_enc, prot_enc = self.tokenize_batch([smiles], [prot_sequence])
        return {k: v.squeeze(0) for k,v in mol_enc.items()}, {k: v.squeeze(0) for k,v in prot_enc.items()}