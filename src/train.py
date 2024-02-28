import argparse
from functools import cache
import shutil
from fsspec.implementations.local import LocalFileSystem
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from dvclive import Live
from dvclive.lightning import DVCLiveLogger


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

def train(resume=False):

    print("Training Random Forest model - START")
    
    transform = transforms.ToTensor()
    train_set = MNIST(root="data", download=True, train=True, transform=transform)
    validation_set = MNIST(root="data", download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set)
    validation_loader = torch.utils.data.DataLoader(validation_set)

    # 
    model = LitAutoEncoder(encoder_size=64, lr=0.01)
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints/',
        filename='mnist-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=-1,
    )

    # trainer = pl.Trainer(
    #     limit_train_batches=200,
    #     limit_val_batches=100,
    #     max_epochs=5,
    #     logger=DVCLiveLogger(log_model=True, save_dvc_exp=False, dvcyaml=False),
    #     callbacks=[checkpoint_callback]
    # )

    fs = LocalFileSystem()
    CKPT_PATH=None
    if resume and fs.exists("models/checkpoints/last.ckpt"):
        CKPT_PATH="models/checkpoints/last.ckpt"
        print("Resuming from checkpoint: ", CKPT_PATH)

    with Live(save_dvc_exp=False, dvcyaml=True) as live:

        trainer = pl.Trainer(
            limit_train_batches=200,
            limit_val_batches=100,
            max_epochs=5,
            logger=DVCLiveLogger(log_model=False, experiment=live),
            callbacks=[checkpoint_callback]
        )
        trainer.fit(
            model, 
            train_loader, 
            validation_loader,
            ckpt_path=CKPT_PATH
        )

        # Log additional metrics after training
        model_path = checkpoint_callback.best_model_path
        # best_model_dst = "dvclive/artifacts/best_model.ckpt"
        best_model_dst = "models/model.ckpt"
        shutil.copy(model_path, best_model_dst)
        live.log_artifact(best_model_dst, name="best", type="model", copy=False)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train your model.')
    parser.add_argument('--resume', action='store_true', help='Resume training if True, start new training otherwise (default: False)')
    args = parser.parse_args()
    
    # Call train function with the value of resume argument
    train(resume=args.resume)
