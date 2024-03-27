import wandb
import random

# start a new wandb run to track this script
base_lr = 0.05
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "base_learning_rate": base_lr,
    "architecture": "Transformer",
    "dataset": "CIFAR-100",
    "epochs": 50,
    }
)

# simulate training
epochs = 50
offset = random.random() / 10 
lr = base_lr
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    if epoch % 10 == 0:
        lr = base_lr + 1 ** -epoch
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss, "lr": lr})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()