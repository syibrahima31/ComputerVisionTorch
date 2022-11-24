import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms

download = False
if download:
    train_dataset = MNIST(root="./", train=True, download=download, transform=transforms.ToTensor())
    test_dataset = MNIST(root="./", train=False, download=download, transform=transforms.ToTensor())
else:
    train_dataset = MNIST(root="./", train=True, download=download, transform=transforms.ToTensor())
    test_dataset = MNIST(root="./", train=False, download=download, transform=transforms.ToTensor())

BATCH_SIZE = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

train_iterator = train_loader.__iter__()
data, target = train_iterator.__next__()

for i, image in enumerate(data):
    image = image.view(28, 28)
    plt.subplot(10, 10, i + 1)
    # plt.imshow(image, cmap='gray')
    plt.axis("off")

# plt.show()

class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10):
        super(NeuralNetworkClassifier, self).__init__()
        self.lin1 = torch.nn.Linear(in_features, 100)
        self.act1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, 100)
        self.act2 = torch.nn.ReLU()
        self.lin3 = torch.nn.Linear(100, out_features)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return x


model = NeuralNetworkClassifier()

# define criterion and optimizer
lr = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#
n_epochs = 100
n_itteration = len(train_loader)

for epoch in range(n_epochs):
    for i, (image, target) in enumerate(train_loader):
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

with torch.no_grad():
    n_total = 0
    n_true = 0
    for image, target in test_loader:
        n = len(image)
        image = image.view(-1, 784)
        # prediction
        pred = model(image)
        pred_target = torch.max(pred, 1)[1]
        good_pred = torch.sum((target == pred_target))
        n_total += n
        n_true += good_pred
    print("Accuracy : ", n_true / n_total)
