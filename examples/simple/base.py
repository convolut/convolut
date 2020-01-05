import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from convolut import Runner, StateManager, MetricManager
from convolut.loader import TrainLoader, ValidLoader
from convolut.logger import ConsoleLogger
from convolut.logger import FileLogger
from convolut.logger.telegram import TelegramLogger
from convolut.logger.tensorboard import TensorboardLogger
from convolut.metric import LossMetric
from convolut.model import ModelManager
from convolut.state import FileCheckpoint
from convolut.trigger.early_stopper import EarlyStopper


# MODEL
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# PYTORCH DATA LOADERS
train_dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                              ])),
                                               batch_size=2, shuffle=False)

# CONVOLUT LOADERS
train_loader = TrainLoader(dataloader=train_dataloader)
valid_loader = ValidLoader(dataloader=train_dataloader)

# OPTIMIZATIONS
device = 'cpu'

model = Net()
model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = Adam(model_params, lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, 4, 1e-6)

# CRITERION
criterion = F.nll_loss

# RUNNER INITIALIZATION AND RUN
epochs = 10
(
    Runner(loaders=[train_loader, valid_loader], epochs=epochs, steps_per_epoch=10)
        # append model training module
        .add(ModelManager(model=model,
                          device=device,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          input_fn=lambda batch: batch[0],
                          target_fn=lambda batch: batch[1]))
        # append state saving/loading
        .add(StateManager())
        .add(FileCheckpoint())
        #  append metric calculation/aggregation
        .add(MetricManager())
        .add(LossMetric())
        # append various triggers
        .add(EarlyStopper())
        # append various loggers
        .add(ConsoleLogger())
        .add(FileLogger())
        .add(TensorboardLogger())
        .add(TelegramLogger(token=os.environ.get("TOKEN"), chat_id=os.environ.get("CHAT_ID")))

        .start()
)
