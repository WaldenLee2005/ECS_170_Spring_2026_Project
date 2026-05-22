import os
import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# If running from root folder use: python -m local_code.stage_4_code.RunStage4

# =========================
# Settings
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CLASS_DATA_DIR = "data/stage_4_data/text_classification"
GEN_DATA_PATH = "data/stage_4_data/text_generation/data"

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 20000
MAX_REVIEW_LEN = 300
BATCH_SIZE = 64

CLASS_EPOCHS = 8
GEN_EPOCHS = 30

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1

GEN_SEQ_LEN = 3
GEN_MAX_WORDS = 15


# =========================
# Text cleaning
# =========================

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are",
    "was", "were", "be", "been", "being", "to", "of", "in", "on", "for",
    "with", "as", "by", "at", "from", "this", "that", "these", "those",
    "it", "its", "i", "you", "he", "she", "they", "we", "me", "him", "her",
    "them", "my", "your", "his", "their", "our", "so", "very", "too"
}


def clean_text(text, remove_stopwords=False):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]

    return words


# =========================
# Vocabulary
# =========================

def build_vocab(token_lists, max_vocab_size=20000):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_tokens(tokens, vocab, max_len):
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


# =========================
# Classification Dataset
# =========================

def load_classification_split(split):
    texts = []
    labels = []

    for label_name, label in [("neg", 0), ("pos", 1)]:
        folder = os.path.join(CLASS_DATA_DIR, split, label_name)

        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                tokens = clean_text(text, remove_stopwords=False)
                texts.append(tokens)
                labels.append(label)

    return texts, labels


class ReviewDataset(Dataset):
    def __init__(self, token_lists, labels, vocab):
        self.x = [encode_tokens(tokens, vocab, MAX_REVIEW_LEN) for tokens in token_lists]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# =========================
# RNN Classification Model
# =========================

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, cell_type="rnn"):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        if cell_type == "rnn":
            self.rnn = nn.RNN(
                EMBED_DIM,
                HIDDEN_DIM,
                NUM_LAYERS,
                batch_first=True,
                bidirectional=True,
                nonlinearity="tanh"
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                EMBED_DIM,
                HIDDEN_DIM,
                NUM_LAYERS,
                batch_first=True,
                bidirectional=True
            )
        elif cell_type == "gru":
            self.rnn = nn.GRU(
                EMBED_DIM,
                HIDDEN_DIM,
                NUM_LAYERS,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise ValueError("cell_type must be rnn, lstm, or gru")

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded)

        if isinstance(hidden, tuple):
            hidden = hidden[0]   # LSTM returns (hidden, cell)

        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]

        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        combined = self.dropout(combined)

        logits = self.fc(combined)
        return logits


def train_classifier(cell_type, train_loader, test_loader, vocab_size):
    model = RNNClassifier(vocab_size, cell_type).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_losses = []
    train_accs = []

    for epoch in range(CLASS_EPOCHS):
        model.train()
        total_loss = 0
        preds = []
        actuals = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(actuals, preds)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        print(f"{cell_type.upper()} Classification Epoch {epoch+1}/{CLASS_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {train_acc:.4f}")

    model.eval()
    test_preds = []
    test_actuals = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            batch_preds = torch.argmax(logits, dim=1)

            test_preds.extend(batch_preds.cpu().numpy())
            test_actuals.extend(y_batch.numpy())

    acc = accuracy_score(test_actuals, test_preds)
    prec = precision_score(test_actuals, test_preds, zero_division=0)
    rec = recall_score(test_actuals, test_preds, zero_division=0)
    f1 = f1_score(test_actuals, test_preds, zero_division=0)
    cm = confusion_matrix(test_actuals, test_preds)

    plt.figure()
    plt.plot(range(1, CLASS_EPOCHS + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{cell_type.upper()} Classification Loss")
    plt.savefig(f"{PLOTS_DIR}/{cell_type}_classification_loss.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, CLASS_EPOCHS + 1), train_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title(f"{cell_type.upper()} Classification Accuracy")
    plt.savefig(f"{PLOTS_DIR}/{cell_type}_classification_accuracy.png")
    plt.close()

    return {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "losses": train_losses,
        "accs": train_accs
    }


# =========================
# Text Generation Dataset
# =========================

def load_jokes():
    with open(GEN_DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    jokes = [clean_text(line, remove_stopwords=False) for line in lines]
    jokes = [joke for joke in jokes if len(joke) > GEN_SEQ_LEN]

    return jokes


class JokeDataset(Dataset):
    def __init__(self, jokes, vocab):
        self.samples = []

        for joke in jokes:
            ids = [vocab.get(w, vocab["<UNK>"]) for w in joke]

            for i in range(len(ids) - GEN_SEQ_LEN):
                x = ids[i:i + GEN_SEQ_LEN]
                y = ids[i + GEN_SEQ_LEN]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, cell_type="rnn"):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        if cell_type == "rnn":
            self.rnn = nn.RNN(
                EMBED_DIM,
                HIDDEN_DIM,
                NUM_LAYERS,
                batch_first=True
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                EMBED_DIM,
                HIDDEN_DIM,
                NUM_LAYERS,
                batch_first=True
            )
        elif cell_type == "gru":
            self.rnn = nn.GRU(
                EMBED_DIM,
                HIDDEN_DIM,
                NUM_LAYERS,
                batch_first=True
            )
        else:
            raise ValueError("cell_type must be rnn, lstm, or gru")

        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(last_output)

        return logits


def train_generator(cell_type, gen_loader, vocab_size):
    model = RNNGenerator(vocab_size, cell_type).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(GEN_EPOCHS):
        model.train()
        total_loss = 0

        for x_batch, y_batch in gen_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(gen_loader)
        losses.append(avg_loss)

        print(f"{cell_type.upper()} Generation Epoch {epoch+1}/{GEN_EPOCHS} | Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, GEN_EPOCHS + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{cell_type.upper()} Text Generation Loss")
    plt.savefig(f"{PLOTS_DIR}/{cell_type}_generation_loss.png")
    plt.close()

    return model, losses


def generate_text(model, start_words, vocab, idx_to_word, max_words=25):
    model.eval()

    words = clean_text(start_words, remove_stopwords=False)

    if len(words) < 3:
        raise ValueError("Please provide at least three starting words.")

    words = words[:3]

    for _ in range(max_words):
        x_ids = [vocab.get(w, vocab["<UNK>"]) for w in words[-GEN_SEQ_LEN:]]
        x_tensor = torch.tensor([x_ids], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)

            next_id = torch.multinomial(probs, num_samples=1).item()
            next_word = idx_to_word.get(next_id, "<UNK>")

        if next_word in ["<PAD>", "<UNK>"]:
            continue

        words.append(next_word)

    return " ".join(words)


# =========================
# Main
# =========================

def main():
    print("Using device:", DEVICE)

    # -------------------------
    # Text classification
    # -------------------------
    print("\nLoading classification data...")

    train_tokens, train_labels = load_classification_split("train")
    test_tokens, test_labels = load_classification_split("test")

    class_vocab = build_vocab(train_tokens, MAX_VOCAB_SIZE)

    print("Train reviews:", len(train_tokens))
    print("Test reviews:", len(test_tokens))
    print("Classification vocab size:", len(class_vocab))

    train_dataset = ReviewDataset(train_tokens, train_labels, class_vocab)
    test_dataset = ReviewDataset(test_tokens, test_labels, class_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_results = {}

    for cell_type in ["rnn", "lstm", "gru"]:
        result = train_classifier(cell_type, train_loader, test_loader, len(class_vocab))
        class_results[cell_type] = result

    # -------------------------
    # Text generation
    # -------------------------
    print("\nLoading joke generation data...")

    jokes = load_jokes()
    gen_vocab = build_vocab(jokes, MAX_VOCAB_SIZE)
    idx_to_word = {idx: word for word, idx in gen_vocab.items()}

    print("Number of jokes:", len(jokes))
    print("Generation vocab size:", len(gen_vocab))

    gen_dataset = JokeDataset(jokes, gen_vocab)
    gen_loader = DataLoader(gen_dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen_results = {}

    start_examples = [
        "what did the",
        "why did the",
        "how many people",
        "what do you"
    ]

    for cell_type in ["rnn", "lstm", "gru"]:
        model, losses = train_generator(cell_type, gen_loader, len(gen_vocab))

        generated_examples = []
        for start in start_examples:
            text = generate_text(model, start, gen_vocab, idx_to_word, GEN_MAX_WORDS)
            generated_examples.append((start, text))

        gen_results[cell_type] = {
            "model": model,
            "losses": losses,
            "examples": generated_examples
        }

    # -------------------------
    # Save results
    # -------------------------
    result_path = os.path.join(RESULTS_DIR, "stage4_results.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Stage 4 RNN Text Classification and Generation Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Text Classification Results\n")
        f.write("-" * 60 + "\n")

        for cell_type, result in class_results.items():
            f.write(f"\nModel: {cell_type.upper()}\n")
            f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall:    {result['recall']:.4f}\n")
            f.write(f"F1 Score:  {result['f1']:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(result["confusion_matrix"]) + "\n")

        f.write("\n\nText Generation Results\n")
        f.write("-" * 60 + "\n")

        for cell_type, result in gen_results.items():
            f.write(f"\nModel: {cell_type.upper()}\n")
            f.write(f"Final Generation Loss: {result['losses'][-1]:.4f}\n")

            for start, text in result["examples"]:
                f.write(f"\nStarting words: {start}\n")
                f.write(f"Generated text: {text}\n")

    print("\nDone.")
    print("Results saved to:", result_path)
    print("Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()