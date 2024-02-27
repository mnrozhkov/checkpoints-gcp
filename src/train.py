import pandas as pd
import joblib
import random
from sklearn.ensemble import RandomForestClassifier
import time


from dvclive import Live

import lightning.pytorch as pl
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder_size=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, encoder_size),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_size, 3)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, encoder_size),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_size, 28 * 28)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        train_mse = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_mse", train_mse)
        return train_mse

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_mse = torch.nn.functional.mse_loss(x_hat, x)
        self.log("val_mse", val_mse)
        return val_mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

def train():

    print("Training Random Forest model - START")
    



    transform = transforms.ToTensor()
    train_set = MNIST(root="data/MNIST", download=True, train=True, transform=transform)
    validation_set = MNIST(root="data/MNIST", download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set)
    validation_loader = torch.utils.data.DataLoader(validation_set)

    from dvclive.lightning import DVCLiveLogger
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(dirpath='mnt/disk1/checkpoints/')
    
    for encoder_size in (64, 128):
        for lr in (1e-3, 0.1):
            model = LitAutoEncoder(encoder_size=encoder_size, lr=lr)
            trainer = pl.Trainer(
                limit_train_batches=200,
                limit_val_batches=100,
                max_epochs=5,
                logger=DVCLiveLogger(log_model=True, report="notebook"),
                callbacks=[checkpoint_callback]
            )
            trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    train()
