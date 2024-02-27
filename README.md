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
