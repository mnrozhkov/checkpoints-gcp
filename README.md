# Checkpoints with Pytoch Lightning

## Install

Create virtual environment named `.venv` (you may use other name)

```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run

```bash
dvc repro 
```

### Train with checkpoints

- Run `dvc exp run -s train  -S "train.is_resume=null" -f` to train the model
- During training, checkpoints will be saved to `dvc-cse/checkpoints-gcp/checkpoints`
- Checkpoints are saved every epoch and names as `model_{exp_name}_{epoch}_{metric}.ckpt`
- The best checkpoint will be saved in `models/model.ckpt`

Example

```bash
dvc exp run -s train  -S "train.is_resume=null" -f
```

### Resume from a checkpoint (with GCS as storage backend)

- Find the checkpoint you want to resume from in GCS `dvc-cse/checkpoints-gcp/checkpoints`  
- Run a new DVC exp with the checkpoint path  `dvc exp run -s train -S "train.resume_checkpoint=PATH_TO_CKPT`

Example

```bash
dvc exp run -s train -S "train.resume_checkpoint=gs://dvc-cse/checkpoints-gcp/checkpoints/basic-lats/mnist-basic-lats-02-val_loss0.062.ckpt" -f
```
