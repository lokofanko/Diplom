import os
import sys
import gc
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import yaml
import argparse
import pickle
import lmdb
import random
import glob
import warnings
from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit.Chem import QED
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# ── Глушим шум ────────────────────────────────────────────────
rdBase.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# ── Безопасный импорт SA-score ─────────────────────────────────
try:
    from BRSAScore import SAScorer
    br_sa_scorer = SAScorer()
    print("✅ Используется BR-SAScore для оценки синтетической доступности")
except Exception as e:
    br_sa_scorer = None
    print(f"⚠️ BR-SAScore недоступен ({e}); fallback на RDKit sascorer")

try:
    import sascorer
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import sascorer
    except ImportError:
        print("⚠️ sascorer.py не найден! SA-score будет всегда = 3.0")
        class _Dummy:
            def calculateScore(self, mol): return 3.0
        sascorer = _Dummy()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from moses.models_storage import ModelsStorage
from src.utils.tokenizers import MolProteinTokenizer
from src.architectures.fusion import BindingPredictor

# ── НОВЫЕ ХИМИЧЕСКИЕ ПАТТЕРНЫ (СТРОГИЕ) ─────────────────────────
ZBG_PATTERNS =[
    # Тетразол (надежный паттерн для 5-членного кольца с 4 азотами)
    Chem.MolFromSmarts("[c,n]1[n,nH][n,nH][n,nH][n,nH]1"),
    # Карбоксилат (включая кислоту и анион)
    Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1H0-1]"),
    # Сульфонамид (строго ловит связь и с ароматикой, и с алифатикой)
    Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]-S(=O)(=O)-[#6]"),
    # Тиол
    Chem.MolFromSmarts("[SX2H]"),
]
CYANO_PAT = Chem.MolFromSmarts("[C]#N")

# Проверка паттернов
for i, p in enumerate(ZBG_PATTERNS):
    if p is None: raise ValueError(f"Ошибка в SMARTS паттерне ZBG #{i}")

# ── Утилиты ────────────────────────────────────────────────────
def get_moses_collate_fn(vocab):
    c2i = getattr(vocab, 'c2i', getattr(vocab, 'w2i', None))
    def _tok(keys):
        for k in keys:
            if k in c2i: return c2i[k]
        return None
    bos, eos, pad, unk = _tok(['^', '<bos>']), _tok(['$', '<eos>']), _tok([' ', '<pad>']), _tok(['?', '<unk>'])
    pad_val = pad if pad is not None else 0

    def collate_fn(batch):
        batch =[s for s in batch if s and isinstance(s, str)]
        if not batch: return None, None
        batch.sort(key=len, reverse=True)
        toks, lens = [],[]
        for s in batch:
            t =[]
            if bos is not None: t.append(bos)
            t.extend([c2i.get(c, unk or 0) for c in s])
            if eos is not None: t.append(eos)
            toks.append(torch.tensor(t, dtype=torch.long))
            lens.append(len(t))
        return pad_sequence(toks, batch_first=True, padding_value=pad_val), torch.tensor(lens, dtype=torch.long)
    return collate_fn

def infinite_iter(loader):
    it = iter(loader)
    while True:
        try: yield next(it)
        except StopIteration: it = iter(loader); yield next(it)

class TargetLigandDataset(Dataset):
    def __init__(self, index_path, db_path, split_path, min_affinity=6.0):
        split_df = pd.read_csv(split_path)
        ids = set(split_df['row_id'].tolist())
        idx_df = pd.read_csv(index_path, low_memory=False)
        aff = next((c for c in idx_df.columns if 'pic50' in c.lower() or 'affinity' in c.lower()), None)
        mask = idx_df['row_id'].isin(ids)
        if aff: mask &= (idx_df[aff] >= min_affinity)
        self.row_ids = idx_df[mask]['row_id'].tolist() or [0]
        self.db_path = db_path
        self.env = None
        print(f"✅ В Prior загружено {len(self.row_ids)} лигандов NDM-1.")

    def __len__(self): return len(self.row_ids)
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.db_path, readonly=True, lock=False)
        rid = self.row_ids[idx]
        if rid == 0: return "CC(=O)Oc1ccccc1C(=O)O"
        with self.env.begin() as txn:
            raw = txn.get(f'sample_{rid}'.encode()) or txn.get(f'proc_{rid}'.encode())
            if raw:
                obj = pickle.loads(raw)
                s = obj.get('smiles') or obj.get('SMILES')
                if s: return s
        return "CC(=O)Oc1ccccc1C(=O)O"

# ── ИЗОЛИРОВАННАЯ СРЕДА РЕВАРДОВ ───────────
class RewardEnvironment:
    def __init__(self, config):
        p, t_cfg = config['paths'], config['tokenizer']
        self.target_seq = config['target']['sequence']
        reward_cfg = config.get("reward", {})
        self.target_n_zbg = int(reward_cfg.get("target_n_zbg", 1))
        self.near_miss_zbg_reward = float(reward_cfg.get("near_miss_zbg_reward", 0.05))
        self.predictor_batch_size = int(reward_cfg.get("predictor_batch_size", 32))
        self.keep_predictor_on_gpu = bool(reward_cfg.get("keep_predictor_on_gpu", True))

        self.tokenizer = MolProteinTokenizer(
            p['chem_bert_path'], p['prot_bert_path'],
            t_cfg['mol_max_len'], t_cfg['prot_max_len']
        )

        self.predictor = BindingPredictor(
            base_model_dir=os.path.dirname(p['prot_bert_path']),
            hidden_dim=1024, freeze_encoders=True
        )

        state = torch.load(p['predictor_checkpoint'], map_location='cpu')
        self.predictor.load_state_dict({k.replace("net.", "head."): v for k, v in state.items()}, strict=False)
        self.predictor.eval().cpu()
        self.predictor_device = torch.device("cpu")
        self.sa_cache = {}

    def _calculate_sa(self, smiles, mol):
        cache_key = smiles
        if cache_key in self.sa_cache:
            return self.sa_cache[cache_key]
        try:
            if br_sa_scorer is not None:
                raw_sa_result = br_sa_scorer.calculateScore(smiles)
                raw_sa = raw_sa_result[0] if isinstance(raw_sa_result, (tuple, list)) else raw_sa_result
            else:
                raw_sa = sascorer.calculateScore(mol)
            raw_sa = float(raw_sa)
        except Exception:
            raw_sa = 10.0 # Если не считается, значит структура сильно штрафуется
        self.sa_cache[cache_key] = raw_sa
        return raw_sa

    def get_rewards(self, device, smiles_list, current_step):
        valid_smiles, valid_mols, valid_idx = [], [],[]
        for i, s in enumerate(smiles_list):
            if s and isinstance(s, str) and len(s) < 256:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    valid_smiles.append(s)
                    valid_mols.append(mol)
                    valid_idx.append(i)

        rewards = torch.zeros(len(smiles_list), device='cpu')
        raw_pic50 =[0.0] * len(valid_smiles)

        # Словарь для проброса сырых метрик в main() для сохранения хитов
        stats_map = {}

        if not valid_smiles:
            return rewards, 0.0, {}

        # 1. Предиктор на GPU
        if self.predictor_device != device:
            self.predictor.to(device)
            self.predictor_device = device

        for start in range(0, len(valid_smiles), self.predictor_batch_size):
            chunk = valid_smiles[start:start+self.predictor_batch_size]
            m_in, p_in = self.tokenizer.tokenize_batch(chunk, [self.target_seq]*len(chunk))
            with torch.no_grad():
                m_in = {k: v.to(device) for k, v in m_in.items()}
                p_in = {k: v.to(device) for k, v in p_in.items()}
                preds = self.predictor(p_in, m_in)
                if preds.ndim == 0: preds = preds.unsqueeze(0)
            for j, v in enumerate(preds.cpu().tolist()):
                raw_pic50[start+j] = v
            del m_in, p_in, preds

        if not self.keep_predictor_on_gpu:
            self.predictor.cpu()
            self.predictor_device = torch.device("cpu")
            torch.cuda.empty_cache()

        # 2. Curriculum Learning
        if current_step < 1000:
            w_act, w_qed, w_sa = 0.6, 0.2, 0.2
            zbg_penalty_for_zero = 0.4
        elif current_step < 3000:
            w_act, w_qed, w_sa = 0.5, 0.3, 0.2
            zbg_penalty_for_zero = 0.2
        else:
            w_act, w_qed, w_sa = 0.4, 0.3, 0.3
            zbg_penalty_for_zero = 0.05

        # Обязательная санитаризация и расчет
        for k, (mol, pic50) in enumerate(zip(valid_mols, raw_pic50)):
            try:
                Chem.SanitizeMol(mol)
            except:
                pass # Оставляем как есть, QED и SA накажут кривую структуру

            # Активность
            R_act = max(0.0, min(1.0, (pic50 - 4.0) / 6.0))

            # Строгий ZBG
            n_zbg = 0
            for pat in ZBG_PATTERNS:
                n_zbg += len(mol.GetSubstructMatches(pat))

            # nZBG reward: exact target is best, neighboring ZBG count is weakly allowed.
            if self.target_n_zbg == 0:
                if n_zbg == 0:
                    R_zbg = 1.0
                elif n_zbg == 1:
                    R_zbg = self.near_miss_zbg_reward
                else:
                    R_zbg = 0.0
            elif self.target_n_zbg == 1:
                if n_zbg == 1:
                    R_zbg = 1.0
                elif n_zbg == 2:
                    R_zbg = self.near_miss_zbg_reward
                elif n_zbg == 0:
                    R_zbg = zbg_penalty_for_zero
                else:
                    R_zbg = 0.0
            elif self.target_n_zbg == 2:
                if n_zbg == 2:
                    R_zbg = 1.0
                elif n_zbg == 1:
                    R_zbg = self.near_miss_zbg_reward
                else:
                    R_zbg = 0.0
            else:
                if n_zbg == self.target_n_zbg:
                    R_zbg = 1.0
                elif abs(n_zbg - self.target_n_zbg) == 1:
                    R_zbg = self.near_miss_zbg_reward
                else:
                    R_zbg = 0.0

            # Если ZBG-компонента занулила награду, QED/BR-SAScore уже не влияют
            # на итог и молекула не может пройти hit-фильтр по target_n_zbg.
            if R_zbg == 0.0:
                rewards[valid_idx[k]] = 0.0
                continue

            # QED
            raw_qed = QED.qed(mol)
            R_qed = raw_qed

            # BR-SAScore: более устойчивый drop-in аналог RDKit SA-score.
            raw_sa = self._calculate_sa(valid_smiles[k], mol)
            R_sa = max(0.0, min(1.0, (10.0 - raw_sa) / 9.0))

            # Токсичный фильтр (Штраф за циано)
            n_cyano = len(mol.GetSubstructMatches(CYANO_PAT))
            R_tox = 1.0
            if n_cyano > 2: R_tox = 0.1
            if raw_sa > 6.0: R_tox *= 0.5

            # Итоговая формула
            final_R = R_zbg * R_tox * (w_act*R_act + w_qed*R_qed + w_sa*R_sa)
            rewards[valid_idx[k]] = final_R

            # Сохраняем сырые метрики для фильтрации при записи файла
            stats_map[valid_idx[k]] = {
                'pic50': pic50,
                'zbg': n_zbg,
                'qed': raw_qed,
                'sa': raw_sa,
                'cyano': n_cyano
            }

        return rewards, len(valid_smiles)/len(smiles_list), stats_map

# ── MAIN ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f: config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_n_zbg = int(config.get("reward", {}).get("target_n_zbg", 1))
    print(f"🚀 Старт nZBG VAE+neural predictor. target_n_zbg={target_n_zbg}. Устройство: {device}")

    # ClearML can block long cluster runs on network timeouts; BRSA experiments
    # are tracked through CSV/log files instead.
    logger = None

    v_dir = config['paths']['vae_dir']
    save_dir = config['paths']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    vocab = torch.load(os.path.join(v_dir, 'vocab.pt'), map_location='cpu')
    v_conf = torch.load(os.path.join(v_dir, 'config.pt'), map_location='cpu')
    vae = ModelsStorage().get_model_class('vae')(vocab, v_conf)

    checkpoints = glob.glob(os.path.join(save_dir, "vae_step_*.pt"))
    if checkpoints:
        vae.load_state_dict(torch.load(max(checkpoints, key=os.path.getctime), map_location='cpu'))
    else:
        vae.load_state_dict(torch.load(os.path.join(v_dir, 'model.pt'), map_location='cpu'))
    vae.to(device).train()

    env = RewardEnvironment(config)
    prior_ds = TargetLigandDataset(config['paths']['index_csv'], config['paths']['db_processed'], config['paths']['split_indices'], config['target']['prior_min_pIC50'])
    collate_fn = get_moses_collate_fn(vocab)
    prior_loader = DataLoader(prior_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    prior_stream = infinite_iter(prior_loader)
    optimizer = optim.Adam(vae.parameters(), lr=float(config['training']['learning_rate']))

    g_cfg = config['generation']
    output_file = g_cfg['output_file']
    found_smiles = set()
    if os.path.exists(output_file):
        try: found_smiles.update(pd.read_csv(output_file).iloc[:,0].tolist())
        except: pass
    else:
        with open(output_file, 'w') as f:
            f.write("smiles,final_reward,pred_pIC50,n_zbg,QED,SA_score,target_n_zbg\n")

    diversity_buffer =[]
    step = 0
    pbar = tqdm(total=g_cfg['max_steps'], initial=step, desc="Итерации")

    while len(found_smiles) < g_cfg['target_count'] and step < g_cfg['max_steps']:
        optimizer.zero_grad()

        # --- A. Prior ---
        loss_prior = torch.tensor(0.0, device=device)
        batch_prior, _ = next(prior_stream)
        if batch_prior is not None:
            loss_prior = sum(vae(batch_prior.to(device)))

        # --- B. Генерация ---
        with torch.no_grad():
            gen_smiles = vae.sample(config['training']['batch_size'])

        vae.cpu(); gc.collect(); torch.cuda.empty_cache()

        # --- C. Оценка (Предиктор + Химия) ---
        rewards, valid_ratio, stats_map = env.get_rewards(device, gen_smiles, step)
        vae.to(device).train()

        max_raw_pic50 = 0.0
        if stats_map:
            max_raw_pic50 = max([val['pic50'] for val in stats_map.values()])

        # --- D. RL Логика ---
        loss_rl = torch.tensor(0.0, device=device)
        max_r = 0.0
        valid_idx =[i for i, s in enumerate(gen_smiles) if s and Chem.MolFromSmiles(s)]

        if valid_idx:
            r_v = rewards[valid_idx]
            s_v = [gen_smiles[i] for i in valid_idx]
            uniq_ratio = len(set(gen_smiles)) / max(len(gen_smiles), 1)

            for s, r in zip(s_v, r_v):
                if r.item() > 0.4 and s not in diversity_buffer:
                    diversity_buffer.append(s)
            if len(diversity_buffer) > 1000: diversity_buffer = diversity_buffer[-1000:]

            if uniq_ratio > 0.15:
                top_k = max(1, int(len(valid_idx) * 0.4))
                top_s = [s_v[i] for i in torch.argsort(r_v, descending=True)[:top_k].numpy()]

                if len(diversity_buffer) > 10:
                    top_s = list(set(top_s + random.sample(diversity_buffer, min(10, len(diversity_buffer)))))

                t_batch, _ = collate_fn(top_s)
                if t_batch is not None:
                    loss_rl = sum(vae(t_batch.to(device)))
                max_r = r_v.max().item()

        # --- E. Обновление весов ---
        loss = (config['training']['beta_prior'] * loss_prior) + (config['training']['beta_rl'] * loss_rl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), config['training']['clip_grad'])
        optimizer.step()

        # --- F. Сохранение ХИТОВ (ЖЕСТКИЙ ФИЛЬТР) ---
        min_pic50 = g_cfg['min_pIC50']
        if valid_idx:
            for idx in valid_idx:
                s = gen_smiles[idx]
                r = rewards[idx].item()

                stats = stats_map.get(idx, {'pic50': 0.0, 'zbg': 0, 'qed': 0.0, 'sa': 10.0, 'cyano': 100})
                pic50 = stats['pic50']
                n_zbg = stats['zbg']
                qed = stats['qed']
                sa = stats['sa']
                n_cyano = stats['cyano']

                # ИДЕАЛЬНЫЙ ФИЛЬТР ИЗ ТЗ
                if (s not in found_smiles and
                    pic50 >= min_pic50 and
                    qed > 0.6 and
                    sa < 4.0 and
                    n_zbg == target_n_zbg and
                    n_cyano <= 1):

                    found_smiles.add(s)
                    with open(output_file, 'a') as f:
                        f.write(f"{s},{r:.4f},{pic50:.4f},{n_zbg},{qed:.3f},{sa:.2f},{target_n_zbg}\n")

        # --- G. Логи ---
        if step % 10 == 0 and logger:
            logger.report_scalar("Loss", "Prior", loss_prior.item(), step)
            logger.report_scalar("Loss", "RL", loss_rl.item(), step)
            logger.report_scalar("Reward", "Max_Comb_Reward", max_r, step)
            logger.report_scalar("Reward", "Max_Raw_pIC50", max_raw_pic50, step)
            logger.report_scalar("Stats", "Validity", len(valid_idx)/config['training']['batch_size'], step)

        if step > 0 and step % config['training']['save_every'] == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_step_{step}.pt"))

        step += 1
        pbar.update(1)
        pbar.set_postfix({
            "Hits": len(found_smiles),
            "nZBG": target_n_zbg,
            "Max_pIC50": f"{max_raw_pic50:.2f}",
            "Reward": f"{max_r:.2f}"
        })

        del loss, loss_prior, loss_rl
        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    print(f"✅ Готово! target_n_zbg={target_n_zbg}; найдено lead-like молекул: {len(found_smiles)}")

if __name__ == "__main__": main()
