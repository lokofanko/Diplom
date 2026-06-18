# Predictor-Guided Molecular Design

Research code for protein-ligand affinity prediction and constrained molecular
generation. The project combines a neural binding predictor with a variational
autoencoder (VAE) and uses predicted activity to guide target-specific ligand
generation.

The current case study targets metallo-beta-lactamases, but the pipeline is
configured through protein sequences, checkpoints, datasets, and reward
constraints rather than being tied to a single protein.

## Research idea

The generation loop optimizes several objectives jointly:

- predicted binding activity from a ProtBERT/ChemBERTa fusion model;
- molecular quality through QED;
- synthetic accessibility through BR-SAScore;
- configurable structural constraints such as the number of zinc-binding
  groups (ZBGs);
- simple toxicity and diversity controls.

A curriculum changes the activity, QED, and synthetic-accessibility weights
during fine-tuning. Molecules are written to the hit table only after passing
strict activity, QED, BR-SAScore, ZBG, and cyano-group filters.

## Repository layout

```text
binding_core/
|-- configs/       Experiment and model configuration files
|-- scripts/       Preprocessing, training, inference, and VAE fine-tuning
|-- src/           Tokenization, datasets, collation, and model definitions
|-- requirements.txt
`-- README.md       Detailed legacy pipeline notes
```

Large datasets, pretrained models, checkpoints, experiment outputs, logs, and
generated molecules are intentionally excluded from Git.

## Main components

### Binding predictor

The predictor combines protein and ligand representations:

1. ProtBERT encodes the target protein sequence.
2. ChemBERTa encodes the molecular SMILES string.
3. A neural fusion head predicts binding affinity as pIC50.

Relevant files:

- `binding_core/src/architectures/fusion.py`
- `binding_core/src/utils/tokenizers.py`
- `binding_core/scripts/train_predictor_mlp.py`
- `binding_core/scripts/fusion_inference.py`

### Predictor-guided VAE

The BR-SAScore experiment is implemented in:

- `binding_core/scripts/vae_finetune_mbl_neural_nzbg_brsascore.py`
- `binding_core/configs/finetune_vae_mbl_neural_nzbg_brsa_0.yaml`
- `binding_core/configs/finetune_vae_mbl_neural_nzbg_brsa_1.yaml`
- `binding_core/configs/finetune_vae_mbl_neural_nzbg_brsa_2.yaml`

The three configurations generate molecules with exactly 0, 1, or 2 detected
ZBGs. The launchers run configurations sequentially because two simultaneous
predictor/VAE processes can exceed the memory available on an 8 GB GPU.

## Installation

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
git clone https://github.com/lokofanko/Diplom.git
cd Diplom
python -m venv .venv
source .venv/bin/activate
pip install -r binding_core/requirements.txt
git clone https://github.com/molecularsets/moses.git binding_core/moses
pip install -e binding_core/moses
```

Install PyTorch using the build appropriate for the local CUDA driver when the
default package is unsuitable.

## Required external artifacts

The repository does not contain restricted or large artifacts. Before running
an experiment, update the selected YAML configuration with paths to:

- ProtBERT and ChemBERTa model directories;
- a trained neural binding-predictor checkpoint;
- a pretrained MOSES VAE (`model.pt`, `vocab.pt`, and `config.pt`);
- the processed ligand LMDB, index CSV, and split CSV.

## Running an experiment

From `binding_core/`:

```bash
python scripts/vae_finetune_mbl_neural_nzbg_brsascore.py \
  --config configs/finetune_vae_mbl_neural_nzbg_brsa_0.yaml
```

Run all ZBG regimes sequentially:

```bash
PYTHON_BIN=/path/to/python bash scripts/run_neural_nzbg_brsa_experiments.sh
```

Resume with only the 1-ZBG and 2-ZBG regimes:

```bash
PYTHON_BIN=/path/to/python bash scripts/run_neural_nzbg_brsa_1_2_experiments.sh
```

Generated hit tables are written under `binding_core/data/generated/`, logs
under `binding_core/logs/`, and checkpoints under `binding_core/experiments/`.
These directories are ignored by Git.

## Reproducibility notes

- BR-SAScore values are cached by SMILES during a run.
- QED and BR-SAScore are skipped when the structural reward is already zero.
- The neural predictor remains on GPU by default to avoid repeated transfers.
- ClearML is disabled in the BR-SAScore runner to prevent network timeouts from
  blocking completed cluster jobs.

## Status

This is research software under active development. Generated molecules and
predicted activities require independent chemical and experimental validation.

