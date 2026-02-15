#!/usr/bin/env python
import os
import sys
import argparse
import torch
from rdkit import Chem
from rdkit.Chem import Draw

# Добавляем moses в PYTHONPATH (если нужно)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moses.models_storage import ModelsStorage

MODELS = ModelsStorage()

def smiles_to_mols(smiles_list):
    mols, legends = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        Chem.SanitizeMol(mol)
        mols.append(mol)
        legends.append(s)
    return mols, legends

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/moses_vae/config.pt")
    parser.add_argument("--vocab",  type=str, default="experiments/moses_vae/vocab.pt")
    parser.add_argument("--model",  type=str, default="experiments/moses_vae/model.pt")
    parser.add_argument("--n",      type=int, default=10, help="number of molecules to sample")
    parser.add_argument("--output", type=str, default="moses_vae_samples.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1) Загружаем конфиг и vocab
    print(f"🧠 Loading config from {args.config}")
    config = torch.load(args.config, map_location="cpu")

    print(f"📚 Loading vocab from {args.vocab}")
    vocab = torch.load(args.vocab, map_location="cpu")

    # 2) Создаем модель VAE так же, как в train.py
    VAEClass = MODELS.get_model_class('vae')
    model = VAEClass(vocab, config).to(device)

    print(f"🔮 Loading weights from {args.model}")
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3) Сэмплинг SMILES
    print(f"🧪 Sampling {args.n} molecules...")
    with torch.no_grad():
        # у MOSES‑VAE есть метод sample(n) -> список SMILES
        smiles = model.sample(args.n)

    # 4) RDKit картинка
    mols, legends = smiles_to_mols(smiles)
    if not mols:
        print("❌ No valid molecules were generated.")
        return

    n = min(args.n, len(mols))
    img = Draw.MolsToGridImage(
        mols[:n],
        molsPerRow=min(5, n),
        subImgSize=(250, 250),
        legends=legends[:n],
    )
    img.save(args.output)
    print(f"✅ Saved image to {args.output}")

if __name__ == "__main__":
    main()
