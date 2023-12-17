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
        self.output_size = output_size

        self.W1 = nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True))
        self.b1 = nn.Parameter(torch.randn(hidden_size, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(hidden_size, output_size, requires_grad=True))
        self.b2 = nn.Parameter(torch.randn(output_size, requires_grad=True))

        # TODO: don't hardcode relu?

    def forward(self, x):
        # TODO: can we save some memory here?
        with torch.no_grad():
            self.A1 = x @ self.W1 + self.b1
            self.Z1 = torch.relu(self.A1)
            self.A2 = self.Z1 @ self.W2 + self.b2

        return self.A2

    def backward(self, x, y):
        # manually set gradients.
        Y = F.one_hot(y, num_classes=self.output_size)
        m = x.shape[0]
        dL_dA2 = -1.0 * (Y - self.Z2) / m
        D = -(Y - self.Z2)
        dL_dA2 = D / m
        dL_db2 = torch.mean(D, dim=0)
        dL_dW2 = self.Z1.T @ dL_dA2
        dL_dZ1 = dL_dA2 @ self.W2.T
        dL_dA1 = (dL_dZ1 * (self.A1 > 0)) / m
        dL_db1 = torch.mean(dL_dZ1 * (self.A1 > 0), dim=0)
        dL_dW1 = (x.T @ dL_dA1) / m

        self.W1.grad = dL_dW1
        self.b1.grad = dL_db1
        self.W2.grad = dL_dW2
        self.b2.grad = dL_db2

def make_moving_collate_fn(device):
    def collate_move_to_device(batch):
        inputs, targets = zip(*batch)
        moved_inputs = torch.stack(inputs).to(device)
        moved_targets = torch.tensor(targets).to(device)
        return moved_inputs, moved_targets
    return collate_move_to_device


def train_mnist():
    set_random_seed(6283185)

    # Training hyperparameters
    num_epochs = 50
    batch_size = 32

    # Model hyperparameters
    input_size = 784
    hidden_size = 256
    output_size = 10

    # Optimizer hyperparameters
    lr = 0.05
    momentum = 0.9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    moving_collate = make_moving_collate_fn(device)

    # Load dataset
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.13066,), (0.30811,))])
    data_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms)
    data_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms)

    # train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_train_subset = torch.utils.data.Subset(data_train, range(8092))
    train_loader = torch.utils.data.DataLoader(data_train_subset, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)

    # TODO: use device somehow
    model = MLP(input_size, hidden_size, output_size)
    model.to(device)

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(f'Train examples: {len(data_train)}')
    print(f'Test examples: {len(data_test)}')


    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_avg_loss = 0.0
        for it, (x_mb, y_mb) in enumerate(train_loader):
            optimizer.zero_grad()

            x_mb = x_mb.view(x_mb.size(0), -1)
            logits = model(x_mb)
            loss = F.cross_entropy(logits, y_mb)
            epoch_avg_loss = (epoch_avg_loss * it + loss.item()) / (it + 1)
            model.Z2 = F.softmax(logits, dim=1)
            model.backward(x_mb, y_mb)

            optimizer.step()

        print(f'Epoch average loss: {epoch_avg_loss}')


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