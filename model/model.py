import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, in_channels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=5, stride=2, padding=1) # 3 x 250 x 200
        self.max_pooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)  # 3 x 125 x 100

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )  # 16 x 125 x 100 
        self.residual1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=1)
        self.res_bn1 = nn.BatchNorm2d(16)
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )  # 32 x 63 x 50
        self.residual2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=2)
        self.res_bn2 = nn.BatchNorm2d(num_features=32)  # 32 x 63 x 50

        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )  # 64 x 32 x 25 
        self.residual3= nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=2) # 64 x 32 x 25 
        self.res_bn3 = nn.BatchNorm2d(64)
        self.bottleneck4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )  # 64 x 32 x 25 

        self.bottleneck5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )  # 128 x 16 x 13 
        self.residual5= nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=2) # 128 x 16 x 13 
        self.res_bn5 = nn.BatchNorm2d(128)
        self.bottleneck6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )  # 128 x 16 x 13

        self.bottleneck7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )  # 256 x 8 x 7 
        self.residual7= nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=2) # 256 x 8 x 7 
        self.res_bn7 = nn.BatchNorm2d(256)
        self.bottleneck8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )  # 256 x 8 x 7

        self.global_avg = nn.AvgPool2d(kernel_size=(7,8))

        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=16),
            nn.Linear(in_features=16, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch):
        x = x.to(self.device)
        if(x.dim() == 3):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        else:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = self.conv1(x)
        x = self.max_pooling(x)
        temp = x
        x = self.bottleneck1(x)
        x = x + self.residual1(temp)
        x = self.res_bn1(x)
        temp = x
        x = self.bottleneck2(x)
        x = x + self.residual2(temp)
        x = self.res_bn2(x)
        temp = x
        x = self.bottleneck3(x)
        x = x + self.residual3(temp)
        x = self.res_bn3(x)
        x = self.bottleneck4(x)
        temp = x
        x = self.bottleneck5(x)
        x = x + self.residual5(temp)
        x = self.bottleneck6(x)
        temp = x
        x = self.bottleneck7(x)
        x = x + self.residual7(temp)
        x = self.res_bn7(x)
        x = self.bottleneck8(x)
        x = self.global_avg(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
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