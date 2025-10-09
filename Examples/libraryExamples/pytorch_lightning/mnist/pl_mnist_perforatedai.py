# original code from https://github.com/rubentea16/pl-mnist

import torch
import pytorch_lightning as pl
import os

from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(10)
GPA.pc.set_p_epochs_to_switch(10)
GPA.pc.set_input_dimensions([-1, 0])
GPA.pc.set_history_lookback(1)
GPA.pc.set_max_dendrites(5)
GPA.pc.set_testing_dendrite_capacity(False)

model_path = "."


"""
Internal Model is required because you cant change trainer.model during training
"""


class InternalModel(nn.Module):
    def __init__(self):
        super(InternalModel, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 4)
        self.layer_2 = torch.nn.Linear(4, 8)
        self.layer_3 = torch.nn.Linear(8, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.softmax(x, dim=1)

        return x


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        self.model = InternalModel()
        self.validation_step_outputs = []
        self.totalValCorrects = 0
        self.totalTestCorrects = 0
        self.test_step_outputs = []
        self.epochs = 0

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss)
        pred = logits.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        # Increment how many times it was correct
        self.totalValCorrects += pred.eq(y.view_as(pred)).sum()

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("test_loss", loss)
        pred = logits.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        # Increment how many times it was correct
        self.totalTestCorrects += pred.eq(y.view_as(pred)).sum()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.validation_step_outputs).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        goodEpochs = self.epochs
        # The first epoch is a validation epoch that happens before any training, so don't add the score or the layers won't be initialized.
        if self.epochs != 0:
            self.model, restructured, training_complete = (
                GPA.pai_tracker.add_validation_score(
                    self.totalValCorrects / 5000, self.model
                )
            )
            self.model.to("cuda")
            if training_complete:
                # send the early stop signal by not increaseing the good epochs
                goodEpochs = 0
            elif restructured:
                # This call will reinitialize the optimizers to point to the new model
                self.trainer.strategy.setup(self.trainer)
        self.log("Good Epochs", goodEpochs)
        self.epochs += 1
        print("got total correct val: %d" % self.totalValCorrects)
        self.totalValCorrects = 0
        # Clear the list of step outputs so future scores are only for current epoch
        self.validation_step_outputs = []
        self.test_step_outputs = []
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def on_test_epoch_end(self):
        avg_loss = torch.tensor(self.test_step_outputs).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        print("got total correct test: %d" % self.totalTestCorrects)
        self.totalTestCorrects = 0
        # Clear the list of step outputs so future scores are only for current epoch
        self.test_step_outputs = []
        # this is so off right now because the setting is to 5% better or bust.  trying to just debug why it doesnt test properly at the end.
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        GPA.pai_tracker.set_optimizer(torch.optim.Adam)
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        optimArgs = {"params": self.model.parameters(), "lr": 1e-3}
        schedArgs = {"mode": "min", "patience": 5}
        self.optimizer, self.scheduler = GPA.pai_tracker.setup_optimizer(
            self.model, optimArgs, schedArgs
        )
        # don't actually return the scheduler since PB handles scheduler interanlly
        return [self.optimizer]


# Custom Callbacks
class MyPrintingCallback(pl.Callback):

    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")


# ## Prepare Data


def prepare_data():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST("./data", train=False, transform=transform)

    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    mnist_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    return mnist_train, mnist_val, mnist_test


# ### Get train, validation, test data

train, val, test = prepare_data()

# ### Prepare Data Loader

train_loader, val_loader, test_loader = (
    DataLoader(train, batch_size=64),
    DataLoader(val, batch_size=64, shuffle=False),
    DataLoader(test, batch_size=64, shuffle=False),
)

# ## Train Model

model = LightningMNISTClassifier()

model.model = UPA.initialize_pai(model.model)

# Set Early Stopping to be when the system sends a single "bad epoch" meaning training is complete
early_stopping = EarlyStopping("Good Epochs", mode="max", patience=0)
# saves checkpoints to 'model_path' whenever 'val_loss' has a new min

trainer = pl.Trainer(
    max_epochs=300000,
    profiler="simple",
    callbacks=[early_stopping],
    default_root_dir=model_path,
)  # gpus=1

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# ## TensorBoard

# # copy lightning logs from google drive to local machine
os.environ["lightning_logs"] = model_path + "lightning_logs"

model = model.to("cuda")
trainer.test(model=model, dataloaders=test_loader)

# ## Inference
# PATH = checkpoint_callback.best_model_path
# print(PATH)

inference = model

x = torch.cat((test[0][0], test[1][0], test[2][0]), 0)  # 3 image
x = x.unsqueeze(1).to("cuda")

y = [test[0][1], test[1][1], test[2][1]]

import numpy as np

# Do Prediction
model = model.to("cuda")
logits = inference(x)
print("Prediction :", np.argmax(logits.to("cpu").detach().numpy(), axis=1))
print("Real :", y)
