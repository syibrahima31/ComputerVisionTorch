# chargements des packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# chargement du jeu de donnée
train_dataset = FashionMNIST(root="./", download=True, transform=transforms.ToTensor())
test_dataset = FashionMNIST(root="./", download=False, transform=transforms.ToTensor())

BATCH_SIZE = 100
train_dataloader = DataLoader(train_dataset, batch_size=100)
test_dataloader = DataLoader(test_dataset, batch_size=100)

# plot images
iter_image = iter(train_dataloader)
images, target = next(iter_image)

for i, image in enumerate(images):
    image = image.view(28, 28)
    plt.subplot(10, 10, i + 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
plt.show()


class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10):
        super(NeuralNetworkClassifier, self).__init__()
        self.lin1 = torch.nn.Linear(in_features, 20)
        self.act1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(20, 20)
        self.act2 = torch.nn.ReLU()
        self.lin3 = torch.nn.Linear(20, 20)
        self.act3 = torch.nn.ReLU()
        self.lin4 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = self.lin4(x)
        return x


model = NeuralNetworkClassifier()

# define criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


n_epochs = 100
n_itteration = len(train_dataloader )

for epoch in range(n_epochs):
    for i, (image, target) in enumerate(train_dataloader):
        # forward pass
        image = image.view(-1, 784)
        pred = model(image)
        # loss
        loss = criterion(pred, target)
        # calcul des gradient
        loss.backward()
        # update the weights
        optimizer.step()
        # zero grad
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"epoch={epoch + 1}/{n_epochs} step ={i + 1}/{n_itteration} loss = {loss.item()}")
