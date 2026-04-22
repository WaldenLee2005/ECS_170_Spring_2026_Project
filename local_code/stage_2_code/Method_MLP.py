'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class Method_MLP(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # hyperparameters
        self.max_epoch = 30
        self.learning_rate = 1e-3
        self.batch_size = 128

        # model architecture
        # assumes 784 input features and 10 output classes
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

        self.loss_history = []

    def forward(self, x):
        return self.network(x)

    def train_model(self, X, y):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                logits = self.forward(batch_X)
                loss = loss_function(logits, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(loader.dataset)
            self.loss_history.append(epoch_loss)

            print(f'Epoch {epoch + 1}/{self.max_epoch}, Loss: {epoch_loss:.6f}')

    def test_model(self, X):
        self.eval()

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)

        with torch.no_grad():
            logits = self.forward(X_tensor)
            pred_y = torch.argmax(logits, dim=1)

        return pred_y.cpu().numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test_model(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': np.array(self.data['test']['y']),
            'loss_history': self.loss_history
        }
            