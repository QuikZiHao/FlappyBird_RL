import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, in_channels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.max_pooling = nn.MaxPool2d(kernel_size=5)

        self.linear1 = nn.Linear(61152, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.to(self.device)
        if(x.dim() == 3):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.bn3(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.seed = 999
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        Q_new = reward
        if not done:
            Q_new = reward + self.gamma * self.model(next_state)
        target = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()