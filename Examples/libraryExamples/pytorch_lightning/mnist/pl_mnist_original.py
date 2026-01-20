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

model_path = '.'

class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.layer_1 = torch.nn.Linear(28 * 28, 4)
        self.layer_2 = torch.nn.Linear(4, 8)
        self.layer_3 = torch.nn.Linear(8, 10)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
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

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss)


    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("test_loss", loss)
        
    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.validation_step_outputs).mean()
        validation_step_outputs = []
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def on_test_epoch_end(self):
        avg_loss = torch.tensor(self.test_step_outputs).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]


# Custom Callbacks
class MyPrintingCallback(pl.Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')


# ## Prepare Data

def prepare_data():
        
        
    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True,
                                    transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    return mnist_train, mnist_val, mnist_test


# ### Get train, validation, test data

train, val, test = prepare_data()

# ### Prepare Data Loader

train_loader, val_loader, test_loader = DataLoader(train, batch_size=64), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)

# ## Train Model

model = LightningMNISTClassifier()

# Set Early Stopping
early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
# saves checkpoints to 'model_path' whenever 'val_loss' has a new min
checkpoint_callback = ModelCheckpoint(dirpath=model_path+'mnist_{epoch}-{val_loss:.2f}',
                                                                            monitor='val_loss', mode='min', save_top_k=3)

trainer = pl.Trainer(max_epochs=30, profiler='simple', 
                                         callbacks=[early_stopping,checkpoint_callback],
                                         default_root_dir=model_path) #gpus=1

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# ## TensorBoard

# # copy lightning logs from google drive to local machine
os.environ['lightning_logs'] = model_path+'lightning_logs'

trainer.test(dataloaders=test_loader, ckpt_path=None)

# ## Inference
PATH = checkpoint_callback.best_model_path
print(PATH)

inference = LightningMNISTClassifier.load_from_checkpoint(PATH)

x = torch.cat((test[0][0],test[1][0],test[2][0]), 0) # 3 image
x = x.unsqueeze(1).to('cuda')

y = [test[0][1],test[1][1],test[2][1]]

import numpy as np

# Do Prediction
logits = inference(x)
print('Prediction :',np.argmax(logits.to('cpu').detach().numpy(), axis=1))
print('Real :', y)
