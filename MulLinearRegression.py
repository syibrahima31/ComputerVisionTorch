import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# creation du jeu de données
X, y = make_regression(n_samples=100000, n_features=20, noise=20)
in_features, out_features = X.shape[1], 1

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Transformation du jeu de données en tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)


class Network(nn.Module):
    def __init__(self, in_features, out_features):
        super(Network, self).__init__()
        # first layer
        self.linear_1 = nn.Linear(in_features, 20)
        self.act1 = nn.ReLU()
        # second layer
        self.linear_2 = nn.Linear(20, 20)
        self.act2 = nn.ReLU()
        # thrid  layer
        self.linear_3 = nn.Linear(20, 20)
        self.act3 = nn.ReLU()
        # four layer
        self.linear_4 = nn.Linear(20, out_features)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act1(x)
        x = self.linear_2(x)
        x = self.act2(x)
        x = self.linear_3(x)
        x = self.act3(x)
        x = self.linear_4(x)
        return x


model = Network(in_features=in_features, out_features=out_features)

# define criterion and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# training loop
n_iters = 100

