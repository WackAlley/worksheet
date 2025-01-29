import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import time


batch_size = 512 #128
kwargs = {}
# kwargs = {'num_workers': 1, 'pin_memory': True}
train_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True, **kwargs)


class fully_connected_relu_net1(nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.criterion = loss_function
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512), # x = linear(28*28, 512)
            nn.ReLU(), # x = relu(x)
            nn.Linear(512, 512), # x = linear(512, 512)
            nn.ReLU(), # x = relu(x)
            nn.Linear(512, 10), # x = linear(512, 10)
        )

    def forward(self, x):
        # size: batch_dim, 1, 28, 28
        #x = x.flatten(1)
        #print(x.size())
        x = x.view(-1,784)
        x = self.linear_relu_stack(x)
        return x.softmax(dim=1) # bolzmann distribution


class fully_connected_cnn_net1(nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        self.criterion = loss_function
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d() #zerro out some random 2d channels, warning: decreases learning rate
        self.fc1 = nn.Linear(320, 60) # size nach dropout: 128 20 4 4 ##?? batch size 64??
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #print(x.size())
        x = x.view(-1, 320) # umformatieren erste (batch dimension ignorieren, rest als 320 pixel nebeneinander
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class fully_connected_cnn_net2(nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        self.criterion = loss_function
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d() #zerro out some random 2d channels, warning: decreases learning rate
        self.fc1 = nn.Linear(320, 100) # size nach dropout: 128 20 4 4 ##?? batch size 64??
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #print(x.size())
        x = x.view(-1, 320) # umformatieren erste (batch dimension ignorieren, rest als 320 pixel nebeneinander
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


criterion = F.nll_loss

def train():
    model.train() # training mode
    for batch_id, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        out = model(data)
        loss = model.criterion(out, target)
        loss.backward()
        optimizer.step()
    scheduler.step()

def test():
    model.eval() # Gewichte eingefrohren, nicht mehr lernen
    loss = 0
    correct = 0
    for data, target in test_data:
        #print(data.size())
        out = model(data)
        loss += F.nll_loss(out, target, reduction='sum').item()
        pred = out.data.max(1, keepdim=True)[1] # 1 -> batch dimension ignorieren
        correct += pred.eq(target.data.view_as(pred)).sum().item() # eq - prüft auf gleichheit
    #print('avarage loss: ', loss / len(test_data.dataset))
    #print('accuracy: ', 100. * correct / len(test_data.dataset))
    return 100. * correct / len(test_data.dataset), loss

start = time.time()
num_epochs = 110
losses_dict = {}
accuracy_dict = {}
for model_type in [fully_connected_cnn_net1, fully_connected_cnn_net2, fully_connected_relu_net1]:
    for learning_rate in [0.025, 0.03]:
        for gamma in [0.88, 0.91, 0.93]:
            for momentum in [0.7, 0.8]:
                model = model_type(criterion)
                tag = '\n%s, exp, lr=%s, gamma=%s, momentum=%s' % (type(model).__name__, learning_rate, gamma, momentum)
                print(tag)
                losses_dict[tag] = []
                accuracy_dict[tag] = []

                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma) #verbose=True
                best = 0
                for epoch in range(1, num_epochs + 1):
                    train()
                    accuracy, loss = test()
                    losses_dict[tag].append(loss)
                    accuracy_dict[tag].append(accuracy)
                    if accuracy > best:
                        best = accuracy
                print('Höchste erreichte Genauigkeit: ', best)
                torch.save(model.state_dict(), "nets/%s.pt" % tag)

end = time.time()
print('benötigte Zeit: ', end - start, '\n')

losses_df = pd.DataFrame.from_dict(losses_dict)
accuracy_df = pd.DataFrame.from_dict(accuracy_dict)
losses_df.to_csv("loss.csv")
accuracy_df.to_csv("accuracy.csv")
