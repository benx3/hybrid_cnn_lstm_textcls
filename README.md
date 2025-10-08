# Hybrid CNN + LSTM Text Classification

**Idea:** Run a BiLSTM branch (captures long-range sequential signals) and a multi-kernel 1D CNN branch (strong for local n-gram patterns) **in parallel** on the same token embeddings. Concatenate their pooled outputs and feed a classifier.

This project gives you:
- A clean, modular PyTorch codebase
- Train/eval on at least **two public datasets**: `ag_news` and `imdb` (from 🤗 datasets)
- Reproducible runs with configs
- Saved artifacts: checkpoints, metrics (`metrics.json`), training curves, confusion matrix, and reports
- Simple scripts + commands
- Visualize previous runs via `visualize.py`

---

## 1) Quickstart

### Create Conda env
```bash
conda env create -f env.yml
conda activate hybrid-cnn-lstm
# If you don't have CUDA or want CPU only, edit env.yml (remove cudatoolkit) or install torch cpu wheel.
```

### Install via pip (optional alternative)
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train on AG News
```bash
python src/train.py --config configs/ag_news.yaml
```

### Train on IMDB
```bash
python src/train.py --config configs/imdb.yaml
```

Artifacts will be written under `outputs/<run_name>/` (auto-generated with timestamp unless you set `--run_name`).

### Evaluate a checkpoint
```bash
python src/evaluate.py --checkpoint outputs/<run_name>/model.pt --dataset ag_news --split test
```

### Predict
```bash
python src/predict.py --checkpoint outputs/<run_name>/model.pt --text "This phone has great battery life!"
```

### Visualize a past run
```bash
python src/visualize.py --run_dir outputs/<run_name>
```

This will display/produce: training curves (`loss_acc_curves.png`), confusion matrix (`confusion_matrix.png`), and it will print metrics from `metrics.json`.

---

## 2) Model: Parallel Hybrid (CNN || BiLSTM) → Concat → MLP

- **Shared Embedding**: Learned embedding table over a tokenizer vocabulary (WordPiece from a lightweight 🤗 tokenizer).  
- **Branch A — BiLSTM**: 1–2 layers, bidirectional, global max/mean pooling over time.
- **Branch B — CNN**: 1D convolutions with multiple kernel sizes (e.g., 3/4/5), ReLU + max-over-time pooling.  
- **Concatenate** pooled branch features → Dropout → MLP classifier.

**Note:** While we use a 🤗 tokenizer to get token IDs, the model uses its **own trainable embeddings** (we're not loading BERT weights).

---

## 3) Datasets

Loaded via `datasets`:
- `ag_news` (4 classes)
- `imdb` (binary)

Tokenization handled by a fast WordPiece tokenizer (`bert-base-uncased`) for robust handling of OOV. Sequences are padded/truncated to `max_length` in config.

---

## 4) Project Structure

```
hybrid_cnn_lstm_textcls/
├── configs/
│   ├── ag_news.yaml
│   └── imdb.yaml
├── outputs/
├── scripts/
│   ├── run_ag_news.sh
│   └── run_imdb.sh
├── src/
│   ├── data/datasets.py
│   ├── models/hybrid.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── utils.py
│   └── visualize.py
├── env.yml
├── requirements.txt
└── README.md
```

---

## 5) Reproducibility & Logging

- Set `seed` in config for determinism (as much as PyTorch allows).
- All key metrics saved to `metrics.json` and `training_log.csv`.
- Best checkpoint by validation metric (`val_accuracy`) saved to `model.pt`.

---

## 6) Notes & Tips

- Tweak `max_length`, learning rate, batch size for your hardware.
- For reproducibility on GPU, you may set `torch.use_deterministic_algorithms(True)`, but it can slow training.
- You can plug in different tokenizers or replace embeddings with pretrained vectors if desired.

Happy experimenting!
