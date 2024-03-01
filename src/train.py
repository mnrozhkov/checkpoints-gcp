import argparse
from functools import cache
import os
import re
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

def train(resume_checkpoint: str | None = None):    

    print("Training Random Forest model - START")
    print("Resuming from checkpoint: ", resume_checkpoint)
    
    transform = transforms.ToTensor()
    train_set = MNIST(root="data", download=True, train=True, transform=transform)
    validation_set = MNIST(root="data", download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set)
    validation_loader = torch.utils.data.DataLoader(validation_set)

    # 
    model = LitAutoEncoder(encoder_size=64, lr=0.01)


    DVC_EXP_NAME = os.environ.get('DVC_EXP_NAME', None)
    checkpoint_callback = ModelCheckpoint(
        # dirpath='models/checkpoints/',
        dirpath="gs://dvc-cse/checkpoints-gcp/checkpoints",
        filename=f'mnist-{DVC_EXP_NAME}'+'-{epoch:02d}-val_loss{val_mse:.3f}',
        auto_insert_metric_name=False,
        monitor='val_mse',
        mode='min',
        save_top_k=3,
        save_last=False,
        verbose=True,
    )

    
    import dvc.api
    params = dvc.api.params_show()
    resume_checkpoint = params.get('train').get('resume_checkpoint', None)
    is_resume = True if resume_checkpoint else False


    with Live(save_dvc_exp=True, dvcyaml=True, resume=is_resume) as live:

        trainer = pl.Trainer(
            # default_root_dir="gs://dvc-cse/checkpoints-gcp/checkpoints",
            limit_train_batches=200,
            limit_val_batches=100,
            max_epochs=6,
            logger=DVCLiveLogger(log_model=False, experiment=live),
            callbacks=[checkpoint_callback]
        )
        trainer.fit(
            model, 
            train_loader, 
            validation_loader,
            ckpt_path=resume_checkpoint
        )

        # Log additional metrics after training
        print("Best model path: ", checkpoint_callback.best_model_path)

        import gcsfs
        # fs = LocalFileSystem()
        fs = gcsfs.GCSFileSystem(project='iterative-sandbox')

        if fs.exists(checkpoint_callback.best_model_path):
            
            model_path = checkpoint_callback.best_model_path
            best_model_dst = "models/model.ckpt"
            print(f"Copying checkpoint {model_path} to {best_model_dst}")
            # shutil.copy(model_path, best_model_dst)
            fs.get(model_path, best_model_dst)

            live.log_artifact(
                best_model_dst, 
                name="mnist_LitAutoEncoder", 
                type="model", 
                labels=["mnist", "autoencoder", "lightning"],
                meta={"resumed_from": resume_checkpoint}, 
                cache=False
            )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train your model.')
    parser.add_argument('--resume_checkpoint', help='Path to a checkpoint to resime training', default=None)
    args = parser.parse_args()
    
    # Call train function with the value of resume argument
    train(resume_checkpoint=args.resume_checkpoint)
