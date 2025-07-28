import runpod
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://nextjournal.com/gkoehler/pytorch-mnist

n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

RESULTS_DIR = "/runpod-volume/"

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Apply the rectified linear unit to the first convolutional layer
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Do the same for the second convolutional layer, but randomly
        # drop some neurons. A dropout layer is a regularization technique
        # that prevents co-adaptation of neurons, reduces overfitting,
        # and improves generalization by making the network more robust.
        # It essentially forces the forward pass to use a different subset
        # of neurons so that the network can't over-rely on any single path.
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def classifier():
    print("starting data loader")

    print("train_loader begin")
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/files/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )
    print("train_loader end")

    print("test_loader begin")
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/files/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )
    print("test_loader end")

    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(example_data.shape)

    # fig = plt.figure()
    # for i in range(6):
    #    plt.subplot(2,3,i+1)
    #    plt.tight_layout()
    #    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #    plt.title("Ground Truth: {}".format(example_targets[i]))
    #    plt.xticks([])
    #    plt.yticks([])
    # fig.savefig(fname="out.png")

    network = Net()
    if torch.cuda.is_available():
        network.cuda()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch: int):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(network.state_dict(), f"{RESULTS_DIR}/model.pth")
                torch.save(optimizer.state_dict(), f"{RESULTS_DIR}/optimizer.pth")

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    test()
    for epoch in range(1, n_epochs + 1):
        print(f"starting training epoch {epoch}")
        train(epoch)
        test()


if __name__ == "__main__":
    # runpod.serverless.start({'handler': classifier })
    classifier()
