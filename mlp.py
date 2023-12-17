import random

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# produces (0.13066, 0.30811)
def calculate_mnist_mean_std():
    data_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mean_train = torch.mean(data_train.data.float()) / 255.
    std_train = torch.std(data_train.data.float()) / 255.
    return (mean_train.numpy(), std_train.numpy())


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.randn(hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.randn(output_size, requires_grad=True)

    def forward(self, x):
        with torch.no_grad():
            x = torch.relu(x @ self.W1 + self.b1)
            x = x @ self.W2 + self.b2
        return x

    def backward(self, x):
        # manually set gradients.
        # 
        pass


def train_mnist():
    set_random_seed(6283185)

    # Training hyperparameters
    num_epochs = 5
    batch_size = 32

    # Model hyperparameters
    input_size = 784
    hidden_size = 256
    output_size = 10

    # Optimizer hyperparameters
    lr = 0.01
    momentum = 0.9

    # Load dataset
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.13066,), (0.30811,))])
    data_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms)
    data_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    model = MLP(input_size, hidden_size, output_size)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(f'Train examples: {len(data_train)}')
    print(f'Test examples: {len(data_test)}')


    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for (x_mb, y_mb) in train_loader:
            print(x_mb.shape)
            print(y_mb.shape)

            optimizer.zero_grad()

            x_mb = x_mb.view(x_mb.size(0), -1)
            logits = model(x_mb)
            loss = F.cross_entropy(logits, y_mb)

            S = F.softmax(logits, dim=1)

            # TODO: Compute gradients

            optimizer.step()


    # for epoch in range(num_epochs):
    #     print(f'Epoch {epoch + 1}/{num_epochs}')
    #     # Training loop...
    #     for batch in train_loader:
    #         x, y = batch
    #         x = x.view(x.size(0), -1)
    #         optimizer.zero_grad()
    #         y_pred = model(x)
    #         loss = torch.nn.functional.cross_entropy(y_pred, y)
    #         loss.backward()
    #         optimizer.step()


if __name__ == '__main__':
    train_mnist()