import argparse
import gcsfs
import os
from fsspec.implementations.local import LocalFileSystem
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import dvc.api
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from utils import load_checkpoints_from_gs


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
    
    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_set = MNIST(root="data", download=True, train=True, transform=transform)
    validation_set = MNIST(root="data", download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set)
    validation_loader = torch.utils.data.DataLoader(validation_set)

    # Create a model
    model = LitAutoEncoder(encoder_size=64, lr=0.01)

    # Create a ModelCheckpoint callback
    DVC_EXP_NAME = os.environ.get('DVC_EXP_NAME', None)
    checkpoint_callback = ModelCheckpoint(
        # Save checkpoints to GCS bucket
        dirpath=f"gs://dvc-cse/checkpoints-gcp/checkpoints/{DVC_EXP_NAME}",
        filename=f'mnist-{DVC_EXP_NAME}'+'-{epoch:02d}-val_loss{val_mse:.3f}',
        auto_insert_metric_name=False,
        monitor='val_mse',
        mode='min',
        save_top_k=3,
        save_last=False,
        verbose=True,
    )

    # Load the parameters from the dvc.yaml file
    params = dvc.api.params_show()
    resume_checkpoint = params.get('train').get('resume_checkpoint', None)
    is_resume = True if resume_checkpoint else False

    # Train the model with Live context manager
    with Live(save_dvc_exp=True, dvcyaml=True, resume=is_resume) as live:

        trainer = pl.Trainer(
            limit_train_batches=200,
            limit_val_batches=100,
            max_epochs=6,
            logger=DVCLiveLogger(log_model=False, experiment=live), # Pass "live" context manager to DVCLiveLogger
            callbacks=[checkpoint_callback]
        )

        # Train a model with or without resuming
        trainer.fit(
            model, 
            train_loader, 
            validation_loader,
            ckpt_path=resume_checkpoint # None or path to a checkpoint
        )

        # Log additional metrics after training
        print("Best model path: ", checkpoint_callback.best_model_path)
        
        # Load checkpoints to track with DVC
        load_checkpoints_from_gs(
            project="iterative-sandbox", 
            source_dir=f"dvc-cse/checkpoints-gcp/checkpoints/{DVC_EXP_NAME}", 
            dst_dir="models/checkpoints")

        # Track the best model with DVC
        fs = gcsfs.GCSFileSystem(project='iterative-sandbox')
        if fs.exists(checkpoint_callback.best_model_path):
            
            model_path = checkpoint_callback.best_model_path
            best_model_dst = "models/model.ckpt"
            print(f"Copying checkpoint {model_path} to {best_model_dst}")
            fs.get(model_path, best_model_dst)
            fs.get(model_path, "models/model_test.ckpt")

            # Log the best model with DVCLive (automatic registering in DVC Studio)
            live.log_artifact(
                best_model_dst, 
                name="mnist_LitAutoEncoder", 
                type="model", 
                labels=["mnist", "autoencoder", "lightning"],
                meta={"resumed_from": resume_checkpoint}, 
                cache=False
            )

            live.log_artifact(
                "models/model_test.ckpt", 
                name="mnist_model_test", 
                type="model", 
                labels=["mnist", "autoencoder", "lightning"],
                meta={"resumed_from": resume_checkpoint}, 
                cache=True
            )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train your model.')
    args = parser.parse_args()
    
    # Call train function with the value of resume argument
    train()
