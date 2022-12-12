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


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first couche de convolution
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=(3, 3))
        self.pool1 = torch.nn.MaxPool2d((2,2))
        
        # second couche de convolution
        self.conv2 = torch.nn.Conv2d(5, 10, kernel_size=(6, 6))
        self.pool2 = torch.nn.MaxPool2d((2, 2))

        # fully connected



print(data.shape)

conv1 = torch.nn.Conv2d(1, 5, kernel_size=(5, 5))

trans1 = conv1(data)
print(trans1.shape)
pool1 = torch.nn.MaxPool2d((2,2))
trans2 = pool1(trans1)
print(trans2.shape)