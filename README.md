# EEG Source Localization with Transformers

Official Implementation of *Transformer Networks Enable Robust Generalization of Source Localization for EEG Measurements*.  
We train transformer networks to localize one or two simultaneously active EEG sources from scalp measurements, using a realistic head model (subject Ernie) and patch-based source spaces at 5 mm and 10 mm resolution.

---

## Installation

Python 3.9 or later is recommended. Creating a virtual environment is also recommended to manage dependencies.

### Install dependencies

Install PyTorch first by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your platform and CUDA version.  Then install the remaining packages:

```bash
pip install pytorch-lightning torchmetrics numpy scipy hdf5storage
```

**Summary of dependencies**

| Package | Purpose |
|---|---|
| `torch` | Neural networks and GPU training |
| `pytorch-lightning` | Training loop, logging, checkpointing |
| `torchmetrics` | Accuracy metrics |
| `numpy` | Array operations |
| `scipy` | MATLAB file I/O |
| `hdf5storage` | HDF5-based `.mat` file loading |

---

## Data

The data directory is not included in the repository due to file size.  Download it from Google Drive and place it at `data/Ernie/` inside the repository root.

**Download link:** https://drive.google.com/drive/folders/1VZ5WJbzCjdDs9v0zgrc9hlQas-ypGb2a?usp=drive_link

After downloading, the directory should have this structure:

```
data/
└── Ernie/
    ├── ernie_eeg_simulations.mat          # Electrode positions, dipole locations/orientations
    ├── leadfield_patch5mmdistance.mat     # Leadfield matrix for 5 mm patch source space
    ├── leadfield_patch10mmdistance.mat    # Leadfield matrix for 10 mm patch source space
    ├── geodesic_distance_matrix_5mm_patches.mat
    ├── geodesic_distance_matrix_10mm_patches.mat
    ├── noise_vectors_5mm/                 # Pre-computed validation and test noise
    │   ├── noise_vectors_val_50.npy
    │   ├── noise_vectors_test_50.npy
    │   └── ...
    └── noise_vectors_10mm/
        ├── noise_vectors_val_50.npy
        ├── noise_vectors_test_50.npy
        └── ...
```

The noise vector files are named by noise level (e.g. `_50` = 50% noise level).  If a file for a given noise level does not exist, it will be generated automatically on the first run and saved for reproducibility.

---

## Project structure

```
EEGFinal/
├── main.py                    # Entry point: training, validation, and testing
├── clusterNetworks.py         # Network architectures
├── myDatasets.py              # PyTorch Dataset classes
├── helpers.py                 # Data loading utilities
├── positional_encoding_3d.py  # 3D sinusoidal positional encoding
└── data/Ernie/                # Downloaded data (see above)
```

---

## Running experiments

All experiments are launched through `main.py`.  The two required positional arguments are the **source space** (`5mm` or `10mm`) and the **noise level** (a fraction between 0 and 1).

```
python main.py <clusters> <noise_level> [options]
```

Training logs are written to `my_logs/` and can be monitored with TensorBoard:

```bash
tensorboard --logdir my_logs
```

The best checkpoint (by validation accuracy) is saved automatically and used for the final test evaluation.

---

### Single-source localization

Predict the cluster index of a single active source.

```bash
# Transformer (default network), 10 mm source space, 50% noise
python main.py 10mm 0.50

# 5 mm source space
python main.py 5mm 0.50
```

---

### Dual-source localization

Predict the cluster indices of two simultaneously active sources.  A permutation-invariant loss is used so the assignment of predictions to sources does not need to be fixed.

Source strengths are mixed with a random ratio α drawn from `[alpha_min, alpha_max]` (default 0.25 – 0.75), simulating sources of unequal amplitude.  In addition to overall accuracy, the following metrics are reported:

- **strong\_acc** — accuracy on the higher-amplitude source (α ≥ 0.5)
- **weak\_acc** — accuracy on the lower-amplitude source (α < 0.5)
- **weighted\_acc** — accuracy weighted by each source's relative amplitude

```bash
# Dual-source, default alpha range [0.25, 0.75]
python main.py 10mm 0.50 --num_sources 2

# Equal-strength sources (alpha fixed at 0.5)
python main.py 10mm 0.50 --num_sources 2 --alpha_range 0.5 0.5

# Wider alpha range (more extreme strength differences)
python main.py 10mm 0.50 --num_sources 2 --alpha_range 0.1 0.9
```

---

### Electrode dropout

Randomly remove `dropout_num` electrodes on every forward pass (training **and** evaluation).  This trains the model to be robust to missing or noisy sensors and reports metrics under that condition.

```bash
# Drop 5 electrodes per sample
python main.py 10mm 0.50 --dropout_num 5

# Dual-source with electrode dropout
python main.py 10mm 0.50 --num_sources 2 --dropout_num 5
```

---

### Using the linear baseline

A three-layer MLP is available as a baseline.  It supports the same `dropout_num` option but only for single-source training.

```bash
python main.py 10mm 0.50 --network_name Linear
python main.py 10mm 0.50 --network_name Linear --dropout_num 5
```

---

### Full option reference

```
usage: main.py [-h] [--network_name {TransformerEncoder,Linear}]
               [--batch_size N] [--max_epochs N] [--learning_rate LR]
               [--scheduler {onecycle,step}]
               [--pe {none,enc}]
               [--workers N] [--no_progress_bar]
               [--num_sources {1,2}]
               [--min_dist MM]
               [--alpha_range MIN MAX]
               [--predict_alpha] [--alpha_weight W]
               [--dropout_num N]
               clusters noise_level

Positional arguments:
  clusters              Source space resolution: '5mm' or '10mm'
  noise_level           Noise level as a fraction, e.g. 0.50

Key options:
  --network_name        TransformerEncoder (default) or Linear
  --num_sources         1 (default) for single-source, 2 for dual-source
  --dropout_num         Electrodes randomly removed per forward pass (default: 0)
  --alpha_range         Strength mixing range for dual-source (default: 0.25 0.75)
  --predict_alpha       Add auxiliary head to predict alpha (dual-source only)
  --alpha_weight        Weight of alpha prediction loss (default: 0.1)
  --min_dist            Minimum geodesic separation (mm) between dual sources (default: 20)
  --pe                  Positional encoding: none (default) or enc
  --batch_size          Default: 512
  --max_epochs          Default: 200
  --learning_rate       Default: 0.0001
  --scheduler           onecycle (default) or step
  --workers             DataLoader worker processes (default: 4)
  --no_progress_bar     Suppress the training progress bar
```

---