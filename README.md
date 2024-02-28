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

- Run `dvc exp run -s train` to train the model
- During training, checkpoints will be saved in the `models/checkpoints` 
- Checkpoints are saved every epoch and names as `model_epoch_{epoch}.ckpt`
- Also, the last checkpoint saved with name `models/last.ckpt`
- The best checkpoint will be saved in `models/model.ckpt`

### Resume from a checkpoint (with GCS as storage backend)

- Download/Update the latest stete of the training pipeline 
  - From GS checkpoints to `models/checkpoints`
  - Script resumes from a checkpoint `models/checkpoints/last.ckpt`
  - Note: assume code version and parameters are the same
- Commit changes to DVC and Git 
  - `dvc commit`
  - `git commit -m "Update checkpoints"`
  - `git push` & `dvc push`
- Run `dvc exp run -s train  -S "train.is_resume=true"` to resume training from the `models/last.ckpt`
