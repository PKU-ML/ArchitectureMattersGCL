# Training based on SimCLR

Here we provide implementation of experiments with SimCLR ([A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)) as backbone.

## Dependencies
* pytorch >=1.2
* torchvision >=0.4.0
* hydra >=0.11.3
* tqdm >=4.45.0

## Code Structure
```pydocstring
models.py           # Define SimCLR model.
simclr.py           # Code to train simclr.
uniform.py          # Code to train on uniform loss.
align.py            # Code to train on align loss.
gaussian.py         # Code to train on gaussian augmentation only.
align_contra_norm.py # Code to train on COntraNorm with align loss.
simclr_lin.py       # Linear Evaluation on frozen SimCLR representations.
contranorm_lin.py   # Linear Evaluation on frozen ContraNorm+Align representations.
simclr_config.py    # Config File with all default hyperparameters in training.
```

## Usage

Train SimCLR :

```bash
python simclr.py
```

Linear evaluation:

```bash
python simclr_lin.py model='YOUR MODEL' fname='OUTPUT MODEL NAME'
```
