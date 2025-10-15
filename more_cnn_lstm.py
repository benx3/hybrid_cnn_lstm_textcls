# train.py
import os, re, bz2, csv, argparse, random, math
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
# 2) Datasets
# ========================
def load_amazon_dataset(train_file="amazon/train.ft.txt.bz2", test_file="amazon/test.ft.txt.bz2"):
    def load_ft(filename):
        data, labels = [], []
        with bz2.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                label, text = line.strip().split(" ", 1)
                labels.append(1 if "__label__2" in label else 0)
                data.append(preprocess_text(text))
        return data, labels
    train_texts, train_labels = load_ft(train_file)
    test_texts,  test_labels  = load_ft(test_file)
    vocab = build_vocab(train_texts + test_texts)
    train_indices = [text_to_indices(t, vocab) for t in train_texts]
    test_indices  = [text_to_indices(t, vocab) for t in test_texts]
    return train_indices, train_labels, test_indices, test_labels, len(vocab)

def load_imdb_dataset(csv_path="imdb/IMDB Dataset.csv"):
    df = pd.read_csv(csv_path)
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    df["tokens"] = df["review"].apply(preprocess_text)
    vocab = build_vocab(df["tokens"])
    df["indices"] = df["tokens"].apply(lambda x: text_to_indices(x, vocab))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    return (train_df["indices"].tolist(), train_df["label"].tolist(),
            test_df["indices"].tolist(),  test_df["label"].tolist(), len(vocab))

def load_agnews_dataset(train_csv="agnews/train.csv", test_csv="agnews/test.csv"):
    train_df = pd.read_csv(train_csv); test_df  = pd.read_csv(test_csv)
    train_df["text"] = train_df["Title"].astype(str) + " " + train_df["Description"].astype(str)
    test_df["text"]  = test_df["Title"].astype(str)  + " " + test_df["Description"].astype(str)
    train_df["label"] = train_df["Class Index"].astype(int) - 1
    test_df["label"]  = test_df["Class Index"].astype(int) - 1
    train_df["tokens"] = train_df["text"].apply(preprocess_text)
    test_df["tokens"]  = test_df["text"].apply(preprocess_text)
    train_df = train_df[train_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    test_df  = test_df [test_df ["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    vocab = build_vocab(list(train_df["tokens"]) + list(test_df["tokens"]))
    train_indices = [text_to_indices(t, vocab) for t in train_df["tokens"]]
    test_indices  = [text_to_indices(t, vocab) for t in test_df["tokens"]]
    return train_indices, train_df["label"].tolist(), test_indices, test_df["label"].tolist(), len(vocab)

def load_dataset(dataset_type="amazon"):
    if dataset_type == "amazon":
        train_idx, train_y, test_idx, test_y, vocab_size = load_amazon_dataset()
        num_classes = 2
    elif dataset_type == "imdb":
        train_idx, train_y, test_idx, test_y, vocab_size = load_imdb_dataset()
        num_classes = 2
    elif dataset_type == "agnews":
        train_idx, train_y, test_idx, test_y, vocab_size = load_agnews_dataset()
        num_classes = 4
    else:
        raise ValueError("dataset_type must be one of: amazon | imdb | agnews")

    train_idx, val_idx, train_y, val_y = train_test_split(
        train_idx, train_y, test_size=0.1, random_state=42, stratify=train_y
    )
    print(f"✅ Loaded {dataset_type}: vocab={vocab_size}, train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx, train_y, val_y, test_y, vocab_size, num_classes

# ========================
# 3) Dataset & Collate
# ========================
class ReviewDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = indices; self.labels = labels
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        x = self.indices[idx]
        if len(x) == 0: x = [1]  # <UNK>
        return torch.tensor(x, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    seqs, labels = zip(*batch)
    fixed = [s if s.numel() > 0 else torch.tensor([1], dtype=torch.long) for s in seqs]
    lengths = torch.tensor([len(s) for s in fixed], dtype=torch.long)
    padded  = pad_sequence(fixed, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(labels)

# ========================
# 4) Embedding helpers
# ========================
def make_pretrained_embedding(embedding_matrix: torch.Tensor, pad_idx: int = 0, freeze: bool = True):
    emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=pad_idx)
    with torch.no_grad(): emb.weight[pad_idx].fill_(0)
    return emb

# ========================
# 5) Models ( với Dropout )
# ========================
class ParallelCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, cnn_out_channels=64, lstm_hidden=128, num_classes=2,
                 embedding_matrix=None, pad_idx=0, freeze_embed=True, dropout=0.3):
        super().__init__()
        if embedding_matrix is not None:
            assert embedding_matrix.shape[1] == embed_dim, "embed_dim không khớp pretrain!"
            self.embedding = make_pretrained_embedding(embedding_matrix, pad_idx, freeze=freeze_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.embed_dropout = nn.Dropout(dropout)

        # CNN branch
        self.conv1 = nn.Conv1d(embed_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.cnn_dropout = nn.Dropout(dropout)

        # LSTM branch
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, dropout=0.0)
        self.lstm_dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(cnn_out_channels + lstm_hidden, num_classes)

    def forward(self, x, lengths):
        embed = self.embed_dropout(self.embedding(x))

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
                 embedding_matrix=None, pad_idx=0, freeze_embed=True, dropout=0.3):
        super().__init__()
        if embedding_matrix is not None:
            assert embedding_matrix.shape[1] == embed_dim, "embed_dim không khớp pretrain!"
            self.embedding = make_pretrained_embedding(embedding_matrix, pad_idx, freeze=freeze_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.embed_dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(embed_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden, batch_first=True, dropout=0.0)
        self.lstm_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x, lengths):
        embed = self.embed_dropout(self.embedding(x))
        cnn_in  = embed.permute(0, 2, 1)                 # (B,E,T)
        cnn_out = torch.relu(self.conv1(cnn_in))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = cnn_out.permute(0, 2, 1)               # (B,T,C)
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

def train_epoch(model, loader, optimizer, criterion, clip_grad_norm=1.0):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for indices, lengths, labels in tqdm(loader, desc="Training"):
        indices = indices.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        optimizer.zero_grad()
        outputs = model(indices, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for indices, lengths, labels in tqdm(loader, desc="Evaluating"):
        indices = indices.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        outputs = model(indices, lengths)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

# ========================
# 7) Main
# ========================
def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-LSTM (Parallel/Sequential) for Text Classification")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dataset", type=str, default="imdb", choices=["amazon", "imdb", "agnews"])
    parser.add_argument("--run_type", type=str, default="parallel", choices=["parallel", "sequence"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--cnn_out", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--pretrained_emb", type=str, default=None, help="path to torch tensor (vocab_size, embed_dim)")
    parser.add_argument("--freeze_embed", type=lambda x: str(x).lower()=='true', default=True)
    parser.add_argument("--pad_idx", type=int, default=0)
    args = parser.parse_args()

    # Load data
    train_idx, val_idx, test_idx, train_y, val_y, test_y, vocab_size, num_classes = load_dataset(args.dataset)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ReviewDataset(train_idx, train_y), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, pin_memory=pin)
    val_loader   = DataLoader(ReviewDataset(val_idx,   val_y), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, pin_memory=pin)
    test_loader  = DataLoader(ReviewDataset(test_idx,  test_y), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, pin_memory=pin)

    # Optional: load pretrained embedding matrix
    embedding_matrix = None
    if args.pretrained_emb:
        embedding_matrix = torch.load(args.pretrained_emb, map_location="cpu")
        if embedding_matrix.dim() != 2:
            raise ValueError("pretrained_emb must be 2-D tensor (vocab_size, embed_dim)")
        if embedding_matrix.shape[1] != args.embed_dim:
            raise ValueError(f"embed_dim ({args.embed_dim}) must match pretrained dim ({embedding_matrix.shape[1]})")
        if embedding_matrix.shape[0] != vocab_size:
            print(f"[Warn] vocab_size ({vocab_size}) != pretrained rows ({embedding_matrix.shape[0]}). "
                  f"Make sure your vocab & embedding align!")

    # Model
    ModelClass = ParallelCNNLSTM if args.run_type == "parallel" else SequentialCNNLSTM
    model = ModelClass(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        cnn_out_channels=args.cnn_out,
        lstm_hidden=args.lstm_hidden,
        num_classes=num_classes,
        embedding_matrix=embedding_matrix,
        pad_idx=args.pad_idx,
        freeze_embed=args.freeze_embed,
        dropout=args.dropout,
    ).to(device)

    # Loss / Optim / Schedulers
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stop = EarlyStopping(mode="min", patience=args.patience, min_delta=args.min_delta)

    # IO paths
    os.makedirs(args.dataset, exist_ok=True)
    best_model_path = f'{args.dataset}/best_model_{args.dataset}_{args.run_type}.pth'
    log_path        = f'{args.dataset}/training_log_{args.dataset}_{args.run_type}.csv'

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
