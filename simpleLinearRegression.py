import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# creation du jeu de données
X, y = make_regression(n_samples=1000, n_features=1, noise=20)

in_features, out_features = X.shape[1], 1

# Transformation du jeu de données en tensor
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

# On trace le nuage de points
plt.scatter(X, y)


# plt.show()

## creation du réseau
class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear(x)
        return x


model = LinearRegression(in_features, out_features)
pred = model(X)

# print(list(model.parameters()))

with torch.no_grad():
    plt.scatter(X, y)
    plt.plot(X, pred)
    # plt.show()

# creterion and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

#
n_iter = 1000

for epoch in range(n_iter):
    # forward pass
    y_pred = model(X)

    # calcul de la fonction perte
    loss = criterion(y_pred, y)

    # calcul des gradients
    loss.backward()

    # update des poids
    optimizer.step()

    #
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"ITTERATION = {epoch + 1}/{n_iter} , LOSS={loss}")

with torch.no_grad():
    pred2 = model(X)
    plt.scatter(X, y)
    plt.plot(X, pred2, c="r")
    plt.show()


