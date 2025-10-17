
# train.py — Hybrid CNN/LSTM with optional BERT (IMDB & AGNews), plus static pretrained embeddings
import os, re, bz2, csv, argparse, random
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Transformers (only needed when --use_bert True)
try:
    from transformers import AutoTokenizer, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# ========================
# 0) Setup
# ========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# 1) Text utils & Vocab
# ========================
def preprocess_text(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        if isinstance(t, list): counter.update(t)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.most_common():
        if f < min_freq: continue
        vocab[w] = len(vocab)
    return vocab

def text_to_indices(tokens, vocab):
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

# ========================
# 2) Datasets (return both indices & raw texts)
# ========================
def load_imdb_dataset(csv_path="imdb/IMDB Dataset.csv"):
    df = pd.read_csv(csv_path)
    # Clean NA
    df = df.dropna(subset=["review", "sentiment"])
    df["label"] = (df["sentiment"].astype(str).str.lower() == "positive").astype(int)
    df["tokens"] = df["review"].apply(preprocess_text)
    vocab = build_vocab(df["tokens"])
    df["indices"] = df["tokens"].apply(lambda x: text_to_indices(x, vocab))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    return (
        train_df["indices"].tolist(), train_df["label"].tolist(), train_df["review"].tolist(),
        test_df["indices"].tolist(),  test_df["label"].tolist(),  test_df["review"].tolist(),
        vocab
    )

def load_agnews_dataset(train_csv="agnews/train.csv", test_csv="agnews/test.csv"):
    train_df = pd.read_csv(train_csv); test_df  = pd.read_csv(test_csv)
    # Expect columns: Class Index, Title, Description
    for col in ["Class Index", "Title", "Description"]:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in AGNews CSVs.")
    train_df["label"] = train_df["Class Index"].astype(int) - 1
    test_df["label"]  = test_df["Class Index"].astype(int) - 1
    train_df["text"] = train_df["Title"].astype(str) + " " + train_df["Description"].astype(str)
    test_df["text"]  = test_df["Title"].astype(str)  + " " + test_df["Description"].astype(str)
    train_df["tokens"] = train_df["text"].apply(preprocess_text)
    test_df["tokens"]  = test_df["text"].apply(preprocess_text)
    # Drop empties
    train_df = train_df[train_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    test_df  = test_df [test_df ["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    vocab = build_vocab(list(train_df["tokens"]) + list(test_df["tokens"]))
    train_indices = [text_to_indices(t, vocab) for t in train_df["tokens"]]
    test_indices  = [text_to_indices(t, vocab) for t in test_df["tokens"]]
    return (
        train_indices, train_df["label"].tolist(), train_df["text"].tolist(),
        test_indices,  test_df["label"].tolist(),  test_df["text"].tolist(),
        vocab
    )

def load_dataset(dataset_type="imdb"):
    if dataset_type == "imdb":
        tr_idx, tr_y, tr_txt, te_idx, te_y, te_txt, vocab = load_imdb_dataset()
        num_classes = 2
    elif dataset_type == "agnews":
        tr_idx, tr_y, tr_txt, te_idx, te_y, te_txt, vocab = load_agnews_dataset()
        num_classes = 4
    else:
        raise ValueError("dataset_type must be one of: imdb | agnews")

    # Split train → train/val while keeping texts & labels aligned
    tr_idx, va_idx, tr_y, va_y, tr_txt, va_txt = train_test_split(
        tr_idx, tr_y, tr_txt, test_size=0.1, random_state=42, stratify=tr_y
    )

    print(f"✅ Loaded {dataset_type}: vocab={len(vocab)}, train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")
    return tr_idx, va_idx, te_idx, tr_y, va_y, te_y, vocab, num_classes, tr_txt, va_txt, te_txt

# ========================
# 3) Dataset & Collate
# ========================
class ReviewDataset(Dataset):
    def __init__(self, xs, labels):
        self.x = xs; self.labels = labels
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        xi = self.x[idx]
        return xi, int(self.labels[idx])

def collate_fn(batch):
    seqs, labels = zip(*batch)
    fixed = []
    for s in seqs:
        if torch.is_tensor(s):
            arr = s
        else:
            arr = torch.tensor(s if isinstance(s, list) else [1], dtype=torch.long)
        fixed.append(arr if arr.numel() > 0 else torch.tensor([1], dtype=torch.long))
    lengths = torch.tensor([len(s) for s in fixed], dtype=torch.long)
    padded  = pad_sequence(fixed, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)

def collate_fn_bert(batch, tokenizer, max_len: int):
    # expects raw strings in x
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    lengths = enc["attention_mask"].sum(dim=1).to(torch.long)
    labels  = torch.tensor(labels, dtype=torch.long)
    return enc, lengths, labels

# ========================
# 4) Embedding helpers (for non-BERT static embeddings)
# ========================
def make_pretrained_embedding(embedding_matrix: torch.Tensor, pad_idx: int = 0, freeze: bool = True):
    emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=pad_idx)
    with torch.no_grad(): emb.weight[pad_idx].fill_(0)
    return emb

def load_pretrained_embedding_matrix(
    path: str,
    vocab: dict,
    embed_dim: int,
    pad_idx: int = 0,
    oov_strategy: str = "random",   # "random" | "zero" | "avg"
    normalize: bool = False,        # L2 normalize vectors
    lowercase_keys: bool = True,
):
    V = len(vocab)
    emb = np.zeros((V, embed_dim), dtype=np.float32)

    rng = np.random.default_rng(42)
    def init_oov():
        if oov_strategy == "zero":
            return np.zeros(embed_dim, dtype=np.float32)
        elif oov_strategy == "avg":
            return None
        else:  # "random"
            limit = np.sqrt(6.0 / (embed_dim + embed_dim))
            return rng.uniform(-limit, limit, size=embed_dim).astype(np.float32)

    vectors = {}
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "tokens" in data and "vectors" in data:
            toks = data["tokens"]; vecs = data["vectors"]
            for t, v in zip(toks, vecs):
                if v.shape[0] != embed_dim: continue
                key = t.lower() if lowercase_keys and isinstance(t, str) else t
                vectors[key] = v.astype(np.float32)
        else:
            obj = dict(data)
            for k, v in obj.items():
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == embed_dim:
                    key = k.lower() if lowercase_keys and isinstance(k, str) else k
                    vectors[key] = v.astype(np.float32)
    elif path.endswith(".pt") or path.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "tokens" in obj and "vectors" in obj:
            toks, vecs = obj["tokens"], obj["vectors"]
            if isinstance(vecs, torch.Tensor): vecs = vecs.cpu().numpy()
            for t, v in zip(toks, vecs):
                if v.shape[0] != embed_dim: continue
                key = t.lower() if lowercase_keys and isinstance(t, str) else t
                vectors[key] = v.astype(np.float32)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor): v = v.cpu().numpy()
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == embed_dim:
                    key = k.lower() if lowercase_keys and isinstance(k, str) else k
                    vectors[key] = v.astype(np.float32)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) <= embed_dim:
                    continue
                word = parts[0]
                try:
                    vec = np.asarray(parts[1:1+embed_dim], dtype=np.float32)
                except ValueError:
                    continue
                if vec.shape[0] != embed_dim: continue
                key = word.lower() if lowercase_keys else word
                vectors[key] = vec

    mean_vec = None
    if oov_strategy == "avg" and len(vectors) > 0:
        mean_vec = np.mean(np.stack(list(vectors.values())), axis=0).astype(np.float32)

    for tok, idx in vocab.items():
        if idx == pad_idx:
            emb[idx] = 0.0; continue
        key = tok.lower() if lowercase_keys else tok
        if key in vectors:
            emb[idx] = vectors[key]
        else:
            emb[idx] = mean_vec if (oov_strategy == "avg" and mean_vec is not None) else init_oov()

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        emb[pad_idx] = 0.0

    return torch.tensor(emb, dtype=torch.float32)

# ========================
# 5) Models (+ BERT option)
# ========================
class BertEmbedding(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", finetune: bool = True):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not installed. Please `pip install transformers`.")
        self.bert = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, enc_batch):
        # enc_batch is a BatchEncoding (has .to())
        out = self.bert(**enc_batch, return_dict=True)
        return out.last_hidden_state  # (B,T,768 for bert-base)

class ParallelCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, cnn_out_channels=64, lstm_hidden=128, num_classes=2,
                 embedding_matrix=None, pad_idx=0, freeze_embed=True, dropout=0.3,
                 use_bert=False, bert_module: nn.Module=None):
        super().__init__()
        self.use_bert = use_bert
        self.bert_module = bert_module

        if not self.use_bert:
            if embedding_matrix is not None:
                assert embedding_matrix.shape[1] == embed_dim, "embed_dim mismatch!"
                self.embedding = make_pretrained_embedding(embedding_matrix, pad_idx, freeze=freeze_embed)
            else:
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            in_channels = embed_dim
        else:
            in_channels = 768  # bert-base
        self.embed_dropout = nn.Dropout(dropout)

        # CNN branch
        self.conv1 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.cnn_dropout = nn.Dropout(dropout)

        # LSTM branch
        self.lstm = nn.LSTM(in_channels, lstm_hidden, batch_first=True, dropout=0.0)
        self.lstm_dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(cnn_out_channels + lstm_hidden, num_classes)

    def forward(self, x, lengths):
        if self.use_bert:
            embed = self.embed_dropout(self.bert_module(x))   # (B,T,768)
        else:
            embed = self.embed_dropout(self.embedding(x))     # (B,T,E)

        # CNN
        cnn_in  = embed.permute(0, 2, 1)                 # (B,E,T)
        cnn_out = torch.relu(self.conv1(cnn_in))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_out = self.pool(cnn_out).squeeze(-1)         # (B,C)
        cnn_out = self.cnn_dropout(cnn_out)

        # LSTM
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        lstm_out = self.lstm_dropout(hn[-1])             # (B,H)

        concat = torch.cat([cnn_out, lstm_out], dim=1)
        return self.fc(concat)

class SequentialCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, cnn_out_channels=64, lstm_hidden=128, num_classes=2,
                 embedding_matrix=None, pad_idx=0, freeze_embed=True, dropout=0.3,
                 use_bert=False, bert_module: nn.Module=None):
        super().__init__()
        self.use_bert = use_bert
        self.bert_module = bert_module

        if not self.use_bert:
            if embedding_matrix is not None:
                assert embedding_matrix.shape[1] == embed_dim, "embed_dim mismatch!"
                self.embedding = make_pretrained_embedding(embedding_matrix, pad_idx, freeze=freeze_embed)
            else:
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            in_channels = embed_dim
        else:
            in_channels = 768  # bert-base

        self.embed_dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden, batch_first=True, dropout=0.0)
        self.lstm_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x, lengths):
        if self.use_bert:
            embed = self.embed_dropout(self.bert_module(x))   # (B,T,768)
        else:
            embed = self.embed_dropout(self.embedding(x))     # (B,T,E)

        cnn_in  = embed.permute(0, 2, 1)                      # (B,E,T)
        cnn_out = torch.relu(self.conv1(cnn_in))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = cnn_out.permute(0, 2, 1)                    # (B,T,C)

        packed = pack_padded_sequence(cnn_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        lstm_out = self.lstm_dropout(hn[-1])
        return self.fc(lstm_out)

# ========================
# 6) Training utils
# ========================
class EarlyStopping:
    def __init__(self, mode="min", patience=5, min_delta=0.0):
        assert mode in ["min", "max"]
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best, self.bad_epochs = None, 0
    def step(self, metric):
        if self.best is None:
            self.best = metric; return False
        improve = (metric < self.best - self.min_delta) if self.mode == "min" else (metric > self.best + self.min_delta)
        if improve:
            self.best = metric; self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience

def _move_to_device(x, device):
    # Handle Hugging Face BatchEncoding (has .to) or dict/tensor
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=True) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x

def train_epoch(model, loader, optimizer, criterion, clip_grad_norm=1.0):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for indices_or_enc, lengths, labels in tqdm(loader, desc="Training"):
        x = _move_to_device(indices_or_enc, device)
        lengths = lengths.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        optimizer.zero_grad()
        outputs = model(x, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for indices_or_enc, lengths, labels in tqdm(loader, desc="Evaluating"):
        x = _move_to_device(indices_or_enc, device)
        lengths = lengths.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        outputs = model(x, lengths)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

# ========================
# 7) Main
# ========================
def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-LSTM (Parallel/Sequential) with optional BERT")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "agnews"])
    parser.add_argument("--run_type", type=str, default="parallel", choices=["parallel", "sequence"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--cnn_out", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--pad_idx", type=int, default=0)

    # Optional static embeddings for NON-BERT
    parser.add_argument("--pretrained_emb", type=str, default=None, help="path to torch tensor (vocab_size, embed_dim)")
    parser.add_argument("--pretrained_path", type=str, default=None, help="path to vectors (txt/npz/pt) to align to vocab")
    parser.add_argument("--freeze_embed", type=lambda x: str(x).lower()=='true', default=True)
    parser.add_argument("--oov_strategy", type=str, default="random", choices=["random","zero","avg"])
    parser.add_argument("--normalize_embed", type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument("--lowercase_pretrain_keys", type=lambda x: str(x).lower()=='true', default=True)

    # BERT
    parser.add_argument("--use_bert", type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_max_len", type=int, default=256)
    parser.add_argument("--bert_finetune", type=lambda x: str(x).lower()=='true', default=True)

    args = parser.parse_args()

    # Load data (both indices & raw texts)
    (train_idx, val_idx, test_idx, train_y, val_y, test_y,
     vocab, num_classes, train_texts, val_texts, test_texts) = load_dataset(args.dataset)

    pin = torch.cuda.is_available()

    # Dataloaders
    if args.use_bert:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not installed. Please `pip install transformers`.")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
        from functools import partial
        train_loader = DataLoader(ReviewDataset(train_texts, train_y), batch_size=args.batch_size,
                                  shuffle=True, collate_fn=partial(collate_fn_bert, tokenizer=tokenizer, max_len=args.bert_max_len),
                                  pin_memory=pin)
        val_loader   = DataLoader(ReviewDataset(val_texts,   val_y), batch_size=args.batch_size,
                                  shuffle=False, collate_fn=partial(collate_fn_bert, tokenizer=tokenizer, max_len=args.bert_max_len),
                                  pin_memory=pin)
        test_loader  = DataLoader(ReviewDataset(test_texts,  test_y), batch_size=args.batch_size,
                                  shuffle=False, collate_fn=partial(collate_fn_bert, tokenizer=tokenizer, max_len=args.bert_max_len),
                                  pin_memory=pin)

        bert_module = BertEmbedding(args.bert_model, finetune=args.bert_finetune).to(device)
        embedding_matrix = None  # not used in BERT mode
    else:
        train_loader = DataLoader(ReviewDataset(train_idx, train_y), batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=pin)
        val_loader   = DataLoader(ReviewDataset(val_idx,   val_y), batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn, pin_memory=pin)
        test_loader  = DataLoader(ReviewDataset(test_idx,  test_y), batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn, pin_memory=pin)

        bert_module = None
        # Optional static embeddings for non-BERT runs
        embedding_matrix = None
        if args.pretrained_emb:
            embedding_matrix = torch.load(args.pretrained_emb, map_location="cpu")
            if embedding_matrix.dim() != 2:
                raise ValueError("pretrained_emb must be 2-D tensor (vocab_size, embed_dim)")
            if embedding_matrix.shape[1] != args.embed_dim:
                raise ValueError(f"embed_dim ({args.embed_dim}) must match pretrained dim ({embedding_matrix.shape[1]})")
            if embedding_matrix.shape[0] != len(vocab):
                print(f"[Warn] vocab_size ({len(vocab)}) != pretrained rows ({embedding_matrix.shape[0]}).")
        elif args.pretrained_path:
            embedding_matrix = load_pretrained_embedding_matrix(
                path=args.pretrained_path,
                vocab=vocab,
                embed_dim=args.embed_dim,
                pad_idx=args.pad_idx,
                oov_strategy=args.oov_strategy,
                normalize=args.normalize_embed,
                lowercase_keys=args.lowercase_pretrain_keys,
            )

    # Build model
    ModelClass = ParallelCNNLSTM if args.run_type == "parallel" else SequentialCNNLSTM
    model = ModelClass(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        cnn_out_channels=args.cnn_out,
        lstm_hidden=args.lstm_hidden,
        num_classes=num_classes,
        embedding_matrix=embedding_matrix,
        pad_idx=args.pad_idx,
        freeze_embed=args.freeze_embed,
        dropout=args.dropout,
        use_bert=args.use_bert,
        bert_module=bert_module,
    ).to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.use_bert:
        # Differential LR: smaller for BERT, larger for classifier/CNN/LSTM
        bert_params = list(model.bert_module.parameters())
        other_params = [p for n, p in model.named_parameters() if "bert_module" not in n]
        optimizer = optim.AdamW([
            {"params": bert_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stop = EarlyStopping(mode="min", patience=args.patience, min_delta=args.min_delta)

    # IO paths
    os.makedirs(args.dataset, exist_ok=True)
    best_model_path = f'{args.dataset}/best_model_{args.dataset}_{args.run_type}.pth'
    if args.use_bert: best_model_path = best_model_path.replace(".pth", "_bert.pth")
    log_path        = f'{args.dataset}/training_log_{args.dataset}_{args.run_type}.csv'
    if args.use_bert: log_path = log_path.replace(".csv", "_bert.csv")

    # Optional: quick debug to verify BERT sees real text
    if args.use_bert:
        _dbg = next(iter(train_loader))
        _enc, _len, _lab = _dbg
        print("\n===== DEBUG BERT BATCH =====")
        print("input_ids shape:", _enc["input_ids"].shape)
        print("lengths sample  :", _len[:8].tolist())
        print("labels sample   :", _lab[:16].tolist())
        # decode one example to ensure it's real English text
        tok = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
        print("\nDecoded sample #0:\n", tok.decode(_enc["input_ids"][0], skip_special_tokens=True)[:300])
        print("===== END DEBUG =====\n")

    # Train/Test
    if args.mode == "train":
        # init CSV
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, clip_grad_norm=args.clip_grad)
            val_loss,   val_acc   = evaluate(model, val_loader, criterion)

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc])

            if val_acc > best_val_acc and args.save_model:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"🔥 Saved best model @ epoch {epoch} (Val Acc {val_acc:.4f})")

            scheduler.step(val_loss)
            if early_stop.step(val_loss):
                print(f"⛔ Early stopping at epoch {epoch}; best val_loss = {early_stop.best:.4f}")
                break

        # Evaluate best on test set if available
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            print(f"🏁 Best Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    else:  # test
        if not os.path.exists(best_model_path):
            print(f"Model file {best_model_path} not found. Please train first."); return
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
