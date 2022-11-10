import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# chargement du jeu de données
df = sns.load_dataset("iris")
df["species"] = df["species"].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

X, y = df.drop(columns=["species"], axis=1).values, df["species"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

in_features, out_features = X_train.shape[1], len(np.unique(y))


######################## Traitement en dataloader ###############

# creation dune classe ToTenser
class ToTensor:
    def __call__(self, samples):
        X, y = samples
        return torch.from_numpy(X), torch.from_numpy(y)


class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.n = len(X)
        self.transform = transform

    def __getitem__(self, item):
        sample = self.X[item], self.y[item]

        if self.transform:
            X, y = self.transform(sample)
            return X, y
        return sample

    def __len__(self):
        return self.n


# creation dune classe qui sappelle irisDatset
dataset_train = IrisDataset(X=X_train, y=y_train, transform=ToTensor())
dataset_test = IrisDataset(X=X_test, y=y_test, transform=ToTensor())

# creation objet dataLoader grace a la classe DataLoader
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=10)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=10)


#################### Creation du modele  #################
class NetworkClassifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(NetworkClassifier, self).__init__()
        self.l1 = torch.nn.Linear(in_features, 10)
        self.act1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(10, out_features)

    def forward(self, x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        return x


model = NetworkClassifier(in_features, out_features)

################### DEfinition dun critere et optimizer #################
lr = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

########## Entrainement par batch ###########################
n_epochs = 100
n_itteration = len(dataloader_train)

for epoch in range(n_epochs):
    for i, (data, target) in enumerate(dataloader_train):

        # forward pass
        pred = model(data.float())

        # calule de la loss
        target = torch.max(target, 1)[1]
        loss = criterion(pred, target)

        # calcul des gradients
        loss.backward()

        # update weights
        optimizer.step()

        # zero grad
        optimizer.zero_grad()

        print(f"epoch= {epoch+1} / {n_epochs}, step = {i+1}/{n_itteration}  loss={loss.item()}")



################ evaluation du modéle ############
with torch.no_grad():
    n_correct = 0
    n_sample = 0
    for data, target in dataloader_test:
        target = torch.max(target, 1)[1]
        pred = model(data.float())
        _, pred_traget = torch.max(pred, 1)
        n_correct += (target==pred_traget).sum()
        n_sample += len(data)
    print( f"Accuracy du jeu de test est: {n_correct/n_sample}")

with torch.no_grad():
    n_correct = 0
    n_sample = 0
    for data, target in dataloader_train:
        target = torch.max(target, 1)[1]
        pred = model(data.float())
        _, pred_traget = torch.max(pred, 1)
        n_correct += (target==pred_traget).sum()
        n_sample += len(data)
    print( f"Accuracy du jeu de train  est: {n_correct/n_sample}")






