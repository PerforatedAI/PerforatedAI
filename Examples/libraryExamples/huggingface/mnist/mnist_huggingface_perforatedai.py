# Original Code from https://discuss.huggingface.co/t/how-to-train-mnist-with-trainer/64960

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import Trainer, TrainingArguments
import numpy as np
import evaluate


from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(200)
# Only used if perforatedbp is installed
# GPA.pc.set_p_epochs_to_switch(200)

GPA.pc.set_output_dimensions([-1, 0, -1, -1])
GPA.pc.set_history_lookback(1)
GPA.pc.set_max_dendrites(5)
GPA.pc.set_testing_dendrite_capacity(False)


class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 2)
        self.conv2 = nn.Conv2d(4, 8, 5, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 10)
        self.act = F.relu

    def forward(self, pixel_values, labels=None):
        x = self.act(self.conv1(pixel_values))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        # FIXED: Return raw logits, not log_softmax
        # The log_softmax will be applied in compute_loss
        return x


device = "cuda"
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
)

train_dset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dset = datasets.MNIST("data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

model = BasicNet()
GPA.metric = "eval_accuracy"
model = UPA.initialize_pai(model)

training_args = TrainingArguments(
    "basic-trainer",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=100000,
    evaluation_strategy="epoch",
    remove_unused_columns=False,
    dataloader_drop_last=False,  # CRITICAL: Don't drop incomplete batches
)


accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    return {"accuracy": accuracy}


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x": pixel_values, "labels": labels}


class MyTrainer(Trainer):
    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        outputs = model(inputs["x"])
        target = inputs["labels"]
        # FIXED: Apply log_softmax here before computing nll_loss
        # Or use cross_entropy which combines softmax + nll_loss
        loss = F.cross_entropy(outputs, target)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to ensure we're getting the right outputs
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(inputs["x"])
            labels = inputs["labels"]
            loss = F.cross_entropy(outputs, labels)

        if prediction_loss_only:
            return (loss, None, None)

        # Return loss, logits, and labels
        # Make sure logits are on CPU for aggregation
        return (loss, outputs.cpu(), labels.cpu())


trainer = MyTrainer(
    model,
    training_args,
    train_dataset=train_dset,
    eval_dataset=test_dset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()
