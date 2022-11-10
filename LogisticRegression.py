import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_iris
import torch.nn.functional as Fun

# chargement du jeu de données
X, y = load_iris(return_X_y=True)


in_features, out_features = X.shape[1], 3

# Transformation du jeu de données en tensor
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
output = Fun.one_hot(y, num_classes=3)
# plt.show()

## creation du réseau
class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear(x)
        return x


model = LogisticRegression(in_features, out_features)
print(model(X))

# print(list(model.parameters()))
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training loop
n_iters = 100
for epoch in range(n_iters):
    # forward pass
    pred = model(X)

    # loss = criterion(pred, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
