from torch.optim.adam import Adam
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from convolut import Runner, StateManager, MetricManager, LoaderName, ScheduleType, FlushType
from convolut.loader import TrainLoader, ValidLoader
from convolut.logger import ConsoleLogger
from convolut.logger.console import ConsoleMode
from convolut.logger.telegram import TelegramMode, TelegramLogger
from convolut.logger.tensorboard import TensorboardMode, TensorboardLogger
from convolut.metric import LossMetric
from convolut.model import ModelManager
from convolut.state import FileCheckpoint
from convolut.trigger.early_stopper import EarlyStopper


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


epochs = 100

train_dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                              ])),
                                               batch_size=4, shuffle=False)

# LOADERS
train_loader = TrainLoader(dataloader=train_dataloader)
valid_loader = ValidLoader(dataloader=train_dataloader)


# OPTIMIZERS AND SCHEDULERS PREPARATION
def optim_fn(model):
    model_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = Adam(model_params, lr=0.0001)
    scheduler = CosineAnnealingLR(optimizer, 4, 1e-6)

    return optimizer, scheduler


# MODEL PREPARATION
device = 'cpu'

model = Net()
optimizer, scheduler = optim_fn(model)

# CRITERION
criterion = F.nll_loss

# RUNNER INITIALIZATION AND RUN
(
    Runner(loaders=[train_loader, valid_loader],
           epochs=epochs,
           steps_per_epoch=1000,
           debug=True)
        # model training
        .add(ModelManager(model=model,
                          device=device,
                          optim_fn=optim_fn,
                          schedule_type=ScheduleType.PerEpoch,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          input_fn=lambda batch: batch[0],
                          target_fn=lambda batch: batch[1]))
        # states
        .add(StateManager())
        .add(FileCheckpoint(folder="run/checkpoints"))
        #  metrics
        .add(MetricManager(flush_type=FlushType.PerLoader))
        .add(LossMetric())
        # triggers
        .add(EarlyStopper(window=3, metric_name="loss", loader_name=LoaderName.Train, delta=1e-1))
        # loggers
        .add(ConsoleLogger(mode=ConsoleMode.SingleLine, folder="run/logs"))
        .add(TensorboardLogger(folder="run/tensorlogs", mode=TensorboardMode.Basic))
        .add(TelegramLogger(token="dfdfsf", channel="tetetet", mode=TelegramMode.Basic))

        .start()
)
