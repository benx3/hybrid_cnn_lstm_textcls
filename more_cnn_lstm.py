import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import bz2
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import csv
import pandas as pd
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# 1️⃣ Load Amazon Reviews
# ========================
def load_amazon_reviews(file_path):
    reviews = []
    labels = []
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for line in enumerate(tqdm(f, desc=f"Loading {file_path}")):
            line = line.strip()
            if line:
                label_text, review_text = line.split(' ', 1)
                label = int(label_text.replace('__label__', '')) - 1
                review_text = re.sub(r'[^\w\s]', '', review_text.lower())
                reviews.append(review_text.split())
                labels.append(label)
    return reviews, labels

# ========================
# 2️⃣ Prepare Data
# ========================
def prepare_data(train_file, test_file):
    train_reviews, train_labels = load_amazon_reviews(train_file)
    test_reviews, test_labels = load_amazon_reviews(test_file)
    
    # Build vocab
    all_words = [word for review in train_reviews + test_reviews for word in review]
    word_counts = Counter(all_words)
    vocab = [word for word, _ in word_counts.most_common()]
    
    word_to_idx = {word: idx + 2 for idx, word in enumerate(vocab)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    def reviews_to_indices(reviews):
        return [[word_to_idx.get(word, 1) for word in review] for review in reviews]
    
    train_indices = reviews_to_indices(train_reviews)
    test_indices = reviews_to_indices(test_reviews)

    return train_indices, train_labels, test_indices, test_labels, len(word_to_idx)

# ========================
# 3️⃣ Dataset & Dataloader
# ========================

# ========== Dataset & Collate ==========
class ReviewDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    indices, labels = zip(*batch)

    # ensure no empty sequences
    fixed = []
    for seq in indices:
        if seq.numel() == 0:  # tensor([])
            # dùng 1 token <UNK> = 1 (vì bạn đã đặt "<UNK>": 1 trong vocab)
            fixed.append(torch.tensor([1], dtype=torch.long))
        else:
            fixed.append(seq)

    lengths = torch.tensor([len(seq) for seq in fixed], dtype=torch.long)
    indices_padded = pad_sequence(fixed, batch_first=True, padding_value=0)
    return indices_padded, lengths, torch.stack(labels)

# ========== Text utilities ==========
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()

# --------------------
# Xây dựng từ vựng
# --------------------
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        if isinstance(text, list):
            counter.update(text)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        vocab[word] = len(vocab)
    return vocab

def text_to_indices(tokens, vocab):
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

# ========== Amazon Reviews ==========
def load_amazon_dataset(train_file="train.ft.txt.bz2", test_file="test.ft.txt.bz2"):
    def load_ft_file(filename):
        data, labels = [], []
        with bz2.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                label, text = line.strip().split(" ", 1)
                labels.append(1 if "__label__2" in label else 0)
                data.append(preprocess_text(text))
        return data, labels

    train_texts, train_labels = load_ft_file(train_file)
    test_texts, test_labels = load_ft_file(test_file)

    vocab = build_vocab(train_texts + test_texts)
    vocab_size = len(vocab)

    train_indices = [text_to_indices(t, vocab) for t in train_texts]
    test_indices = [text_to_indices(t, vocab) for t in test_texts]

    return train_indices, train_labels, test_indices, test_labels, vocab_size

# ========== IMDb ==========
def load_imdb_dataset(csv_path="IMDB Dataset.csv"):
    df = pd.read_csv(csv_path)
    
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    df["tokens"] = df["review"].apply(preprocess_text)

    vocab = build_vocab(df["tokens"])
    vocab_size = len(vocab)
    df["indices"] = df["tokens"].apply(lambda x: text_to_indices(x, vocab))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    return (
        train_df["indices"].tolist(),
        train_df["label"].tolist(),
        test_df["indices"].tolist(),
        test_df["label"].tolist(),
        vocab_size
    )
# --------------------
# Encode thành chỉ số
# --------------------
def encode_text(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

# --------------------
# Xử lý text cơ bản
# --------------------
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ===== Loader cho AG News dataset =====
def load_agnews_dataset(train_csv="train.csv", test_csv="test.csv"):
    import pandas as pd

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

     
    # ================================
    # Làm sạch dữ liệu
    # ================================
    # AG News có cột: ["Class Index", "Title", "Description"]
    # Một số dòng có thể bị NaN, nên ép string và nối lại
    train_df["text"] = train_df["Title"].astype(str).fillna("") + " " + train_df["Description"].astype(str).fillna("")
    test_df["text"] = test_df["Title"].astype(str).fillna("") + " " + test_df["Description"].astype(str).fillna("")

    train_df["label"] = train_df["Class Index"].astype(int) - 1
    test_df["label"] = test_df["Class Index"].astype(int) - 1

    # ================================
    # Tokenization
    # ================================
    train_df["tokens"] = train_df["text"].apply(preprocess_text)
    test_df["tokens"] = test_df["text"].apply(preprocess_text)

    # Loại bỏ dòng lỗi hoặc rỗng
    train_df = train_df[train_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    test_df = test_df[test_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    # ================================
    # Build vocab
    # ================================
    vocab = build_vocab(list(train_df["tokens"]) + list(test_df["tokens"]))
    vocab_size = len(vocab)

    # ================================
    # Convert token → index
    # ================================
    train_indices = [text_to_indices(tokens, vocab) for tokens in train_df["tokens"]]
    test_indices = [text_to_indices(tokens, vocab) for tokens in test_df["tokens"]]
    
    # ================================
    # Lấy label
    # ================================
    train_labels = train_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist()
    
    print(f"[AG News] Loaded {len(train_labels)} train samples, {len(test_labels)} test samples, vocab={vocab_size}")
    print("Label range:", min(train_labels), "->", max(train_labels))

    return train_indices, train_labels, test_indices, test_labels, vocab_size

# ========== Unified Loader ==========
def load_dataset(dataset_type="amazon"):
    if dataset_type == "amazon":
        print("🛒 Loading Amazon Reviews...")
        train_indices, train_labels, test_indices, test_labels, vocab_size = load_amazon_dataset(
            "amazon/train.ft.txt.bz2", "amazon/test.ft.txt.bz2")
    elif dataset_type == "imdb":
        print("🎬 Loading IMDb Reviews...")
        train_indices, train_labels, test_indices, test_labels, vocab_size = load_imdb_dataset(
            "imdb/IMDB Dataset.csv")
    elif dataset_type == "agnews":
        print("📰 Loading AG News...")
        train_indices, train_labels, test_indices, test_labels, vocab_size = load_agnews_dataset(
            "agnews/train.csv", "agnews/test.csv")
    else:
        raise ValueError("dataset_type must be one of: 'amazon', 'imdb', or 'agnews'")

    # Split train/val
    train_indices, val_indices, train_labels, val_labels = train_test_split(
        train_indices, train_labels, test_size=0.1, random_state=42
    )

    print(f"✅ Loaded {dataset_type}: vocab={vocab_size}, train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    return train_indices, val_indices, test_indices, train_labels, val_labels, test_labels, vocab_size

# ========================
# 4️⃣ Models
# ========================

# Parallel Model: CNN and LSTM in parallel, then concat
class ParallelCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, cnn_out_channels=4, lstm_hidden=8, num_classes=2):
        super(ParallelCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # CNN branch
        self.conv1 = nn.Conv1d(embed_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # LSTM branch
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        
        concat_size = cnn_out_channels + lstm_hidden
        self.fc = nn.Linear(concat_size, num_classes)
    
    def forward(self, x, lengths):
        embed = self.embedding(x)
        
        # CNN branch
        cnn_in = embed.permute(0, 2, 1)
        cnn_out = torch.relu(self.conv1(cnn_in))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_out = self.pool(cnn_out).squeeze(-1)
        
        # LSTM branch
        packed_embed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_embed)
        lstm_out = hn[-1]
        
        # Concat
        concat = torch.cat((cnn_out, lstm_out), dim=1)
        out = self.fc(concat)
        return out

# Sequential Model: CNN then LSTM
class SequentialCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, cnn_out_channels=4, lstm_hidden=8, num_classes=2):
        super(SequentialCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # CNN layers (keep sequence length)
        self.conv1 = nn.Conv1d(embed_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        
        # LSTM on CNN output
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden, batch_first=True)
        
        self.fc = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x, lengths):
        embed = self.embedding(x)
        
        # CNN: input (batch, embed_dim, seq_len)
        cnn_in = embed.permute(0, 2, 1)
        cnn_out = torch.relu(self.conv1(cnn_in))
        cnn_out = torch.relu(self.conv2(cnn_out))
        
        # Transpose back to (batch, seq_len, cnn_out_channels) for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # Pack and LSTM
        packed_cnn = pack_padded_sequence(cnn_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_cnn)
        lstm_out = hn[-1]
        
        out = self.fc(lstm_out)
        return out

# ========================
# 5️⃣ Training & Eval
# ========================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for indices, lengths, labels in tqdm(loader, desc="Training"):
        # Nếu DataLoader(pin_memory=True), dùng non_blocking để copy nhanh hơn
        indices = indices.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(indices, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for indices, lengths, labels in tqdm(loader, desc="Evaluating"):
            # Nếu DataLoader(pin_memory=True), dùng non_blocking để copy nhanh hơn
            indices = indices.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            outputs = model(indices, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

# ========================
# 6️⃣ Main
# ========================
def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-LSTM for Amazon Reviews")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Mode: train or test")
    parser.add_argument("--dataset", type=str, default="amazon", choices=["amazon", "imdb", "agnews"],
                        help="Choose dataset: 'amazon', 'imdb' or 'agnews'")
    parser.add_argument('--save_model', action='store_true', default=True, help="Save model after training")
    parser.add_argument('--run_type', type=str, default='parallel', choices=['parallel', 'sequence'], help="Run type: parallel or sequence")
    args = parser.parse_args()

    # ==== Chọn dataset ====
    batch_size = 32
    dataset_type = args.dataset  # 👈 Chọn 'amazon', 'imdb' hoặc 'agnews'

    train_indices, val_indices, test_indices, train_labels, val_labels, test_labels, vocab_size = load_dataset(dataset_type=dataset_type)

    train_loader = DataLoader(ReviewDataset(train_indices, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ReviewDataset(val_indices, val_labels), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ReviewDataset(test_indices, test_labels), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("📦 DataLoader ready!")
    num_classes = 4
    if dataset_type == "agnews":
        num_classes = 4
    else:
        num_classes = 2  # imdb hoặc amazon

    best_model_path = f'{args.dataset}/best_model_{args.dataset}_{args.run_type}.pth'
    # Model selection based on run_type
    if args.run_type == 'parallel':
        model = ParallelCNNLSTM(vocab_size, num_classes=num_classes).to(device)
    elif args.run_type == 'sequence':
        model = SequentialCNNLSTM(vocab_size, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if args.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # File CSV để lưu log
        log_path = f"{args.dataset}/training_log_{args.dataset}_{args.run_type}.csv"

        # Ghi header (chỉ ghi 1 lần)
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        # Training
        num_epochs = 80
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Ghi kết quả vào CSV
            with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

            # Kiểm tra và lưu best model theo val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"🔥 New best model saved at epoch {epoch+1} with Val Acc = {val_acc:.4f}")

    
    elif args.mode == 'test':
        # Load model
        if args.run_type == 'parallel':
            model = ParallelCNNLSTM(vocab_size).to(device)
        elif args.run_type == 'sequence':
            model = SequentialCNNLSTM(vocab_size).to(device)
        try:
            model.load_state_dict(torch.load(best_model_path))
            print(f"Model loaded from {best_model_path}")
        except FileNotFoundError:
            print(f"Model file {best_model_path} not found. Please train first.")
            return
        
        # Test
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()