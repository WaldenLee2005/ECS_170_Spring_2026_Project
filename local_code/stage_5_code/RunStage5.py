import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOCAL_CODE = os.path.join(PROJECT_ROOT, "local_code")

if LOCAL_CODE not in sys.path:
    sys.path.insert(0, LOCAL_CODE)

# Put Dataset_Loader_Node_Classification.py in the same folder as this file.
from Dataset_Loader_Node_Classification import Dataset_Loader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        out = torch.sparse.mm(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def accuracy(output, labels):
    preds = output.argmax(dim=1)
    return (preds == labels).float().mean().item()


def load_dataset(name, data_root):
    loader = Dataset_Loader()
    loader.dataset_name = name
    loader.dataset_source_folder_path = os.path.join(data_root, name)
    return loader.load()


def train_one_dataset(name, data_root, epochs=300, hidden=32, lr=0.01, weight_decay=5e-4, dropout=0.5, use_cuda=False, patience = 30):
    data = load_dataset(name, data_root)
    graph = data['graph']
    split = data['train_test_val']

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    features = graph['X'].to(device)
    labels = graph['y'].to(device)
    adj = graph['utility']['A'].coalesce().to(device)
    idx_train = split['idx_train'].to(device)
    idx_val = split['idx_val'].to(device)
    idx_test = split['idx_test'].to(device)

    model = GCN(
        nfeat=features.shape[1],
        nhid=hidden,
        nclass=int(labels.max().item() + 1),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_loss_val = float("inf")
    best_state = None
    bad_epochs = 0 

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
            acc_val = accuracy(output[idx_val], labels[idx_val])
        
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        history['train_loss'].append(loss_train.item())
        history['val_loss'].append(loss_val)
        history['train_acc'].append(acc_train)
        history['val_acc'].append(acc_val)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'{name:8s} Epoch {epoch+1:03d}: train_loss={loss_train.item():.4f}, train_acc={acc_train:.4f}, val_loss={loss_val:.4f}, val_acc={acc_val:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        test_loss = F.nll_loss(output[idx_test], labels[idx_test]).item()
        test_acc = accuracy(output[idx_test], labels[idx_test])

    os.makedirs('plots', exist_ok=True)
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{name} GCN Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}_loss_early_stopping.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{name} GCN Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}_accuracy_early_stopping.png', dpi=200)
    plt.close()

    return {'dataset': name, 'test_loss': test_loss, 'test_accuracy': test_acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/stage_5_data')
    parser.add_argument('--dataset', default='all', choices=['all', 'cora', 'citeseer', 'pubmed'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    datasets = ['cora', 'citeseer', 'pubmed'] if args.dataset == 'all' else [args.dataset]
    results = []
    for d in datasets:
        results.append(train_one_dataset(d, args.data_root, args.epochs, args.hidden, args.lr, args.weight_decay, args.dropout, args.cuda, args.patience))

    os.makedirs('results', exist_ok=True)
    with open('results/stage5_results_early_stopping.txt', 'w') as f:
        line = f"parameters: epochs={args.epochs}, hidden={args.hidden}, lr={args.lr}, weight_decay={args.weight_decay}, dropout={args.dropout}, cuda={args.cuda}, patience={args.patience}"
        print(line)
        f.write(line + '\n')
        for r in results:
            line = f"{r['dataset']}: test_loss={r['test_loss']:.4f}, test_accuracy={r['test_accuracy']:.4f}"
            print(line)
            f.write(line + '\n')


if __name__ == '__main__':
    main()
