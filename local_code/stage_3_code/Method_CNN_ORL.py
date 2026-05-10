from local_code.base_class.method import method
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Net_ORL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128 * 14 * 11, 256)
        self.fc2 = nn.Linear(256, 40)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Method_CNN_ORL(method):
    data = None
    max_epoch = 50
    batch_size = 50
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        self.net = Net_ORL().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_history = []

    def prepare_data(self, instances):
        X = np.array([inst['image'] for inst in instances], dtype =np.float32)
        y = np.array([inst['label'] for inst in instances], dtype =np.int64)

        X = X / 255.0
        X = X[:, :, :, 0]

        X = X[:, np.newaxis, :, :]

        y=y-1

        return torch.tensor(X), torch.tensor(y)

    def train_model(self):
        X_tensor, y_tensor = self.prepare_data(self.data['train'])
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)
        self.loss_history = []

        for epoch in range(self.max_epoch):
            self.net.train()
            running_loss = 0.0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(inputs), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()*inputs.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            self.loss_history.append(epoch_loss)
            print(f'Epoch {epoch+1}: Loss: {epoch_loss:.6f}')

    def test_model(self):
        X_tensor, y_tensor = self.prepare_data(self.data['test'])
        self.net.eval()

        with torch.no_grad():
            outputs = self.net(X_tensor.to(device))
            pred_y = torch.argmax(outputs, dim=1).cpu().numpy()
        return pred_y, y_tensor.numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model()
        print('--start testing...')
        pred_y, true_y = self.test_model()
        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': self.loss_history
        }