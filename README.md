# Set Prediction without Imposing Structure as Conditional Density Estimation

PyTorch code for the experiments of Deep Energy-based Set Prediction (DESP) and some baselines.
DESP takes an energy-based viewpoint on set prediction and circumvents the necessity for assignment-based set loss training objectives (e.g. Hungarian loss).

For details see [Set Prediction without Imposing Structure as Conditional Density Estimation](https://arxiv.org/abs/2010.04109) by David Zhang, Gertjan Burghouts, and Cees Snoek.

## How to run...
This code base has only been tested on **Python 3.9.1**.
We offer two requirement files listing the **same** packages, for installation via `pip` and `conda` respectively.

### Polygons
```
python run.py -c desp_polygons
python run.py -c baseline_polygons
```

### Digits
```
python run.py -c desp_digits
python run.py -c baseline_digits
```

### Set MNIST Auto-Encoding
```
python run_with_early_stopping.py -c desp_mnist
```

### CLEVR Object Detection
See [DSPN](https://github.com/Cyanogenoid/dspn) for instructions on settings up the CLEVR dataset. Adapt in the config file `configs/desp_clevr.py` the `config.data.base_path` variable accordingly.
```
python run_with_early_stopping.py -c desp_clevr
```

### CelebA Subset Anomaly Detection
```
python run_with_early_stopping.py -c desp_celeba
python run_with_early_stopping.py -c baseline_celeba
```

## Code Structure Overview
The following two main files contain generic code for the training and evaluation procedure, together with the logging logic:
```
run.py
run_with_early_stopping.py
```

The pytorch models in the directory `models` all follow a similar structure and implement experiment specific training and evaluation steps.

The configuration files in `configs` specify the model that is used for training & evaluation, hyperparameters, logging parameters and more.



## BibTeX entry

```
@inproceedings{zhang2021set,
  title     = {Set Prediction without Imposing Structure as Conditional Density Estimation},
  author    = {Zhang, David W and Burghouts, Gertjan J and Snoek, Cees GM},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  eprint    = {2010.04109},
  url       = {https://arxiv.org/abs/2010.04109},
}
```