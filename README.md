# EEG Source Localization with Transformers

Official Implementation of *[Transformer networks are robust learners of source localization for EEG measurements](https://iopscience.iop.org/article/10.1088/1741-2552/aeb82c)*.  
We train transformer networks to localize one or two simultaneously active EEG sources from scalp measurements, using a realistic head model (subject Ernie) and patch-based source spaces at 5 mm and 10 mm resolution.  The repository also contains the multi-source (multi-embedding) comparison against a single-heatmap top-k baseline, the unknown-source-count experiments, and the real-EEG experiments.

---

## Installation

Python 3.9 or later is recommended. Creating a virtual environment is also recommended to manage dependencies.

### Install dependencies

Install PyTorch first by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your platform and CUDA version.  Then install the remaining packages:

```bash
pip install pytorch-lightning torchmetrics tensorboard numpy scipy pandas hdf5storage
```

**Summary of dependencies**

| Package | Purpose |
|---|---|
| `torch` | Neural networks and GPU training |
| `pytorch-lightning` | Training loop, logging, checkpointing |
| `torchmetrics` | Accuracy metrics |
| `tensorboard` | Training logs (`my_logs/`) |
| `numpy` | Array operations |
| `scipy` | MATLAB file I/O, bipartite matching |
| `pandas` | Reading electrode tables in the real-data experiments |
| `hdf5storage` | HDF5-based `.mat` file loading |

**Optional packages**

| Package | Needed for |
|---|---|
| `wandb` | `ablation.py` / `ablation_unknown_count.py` log to [Weights & Biases](https://wandb.ai) by default (`pip install wandb`, then `wandb login`).  It is imported only when used, so it is **not** required if you pass `--tensorboard`. |
| `mne` | `real_data_experiment.py --mode preprocess` only (reading the raw EEGLAB files).  Not needed if you download the preprocessed files (see [Real-data experiments](#real-data-experiments)). |

Tested with Python 3.11, torch 2.3.0, pytorch-lightning 2.2.5, torchmetrics 1.4.0, numpy 1.24, scipy 1.15, pandas 2.2, hdf5storage 0.1.19.  Raw-EEG preprocessing was tested with `mne` 1.9.0 in a separate environment (the two modes exchange data through `.npz` files, so they can live in separate environments or in one).

---

## Data

Nothing in `data/`, `checkpoints/` or `real_data_experiment_output/` is included in the repository due to file size.  Everything is in a single Google Drive folder whose layout mirrors the repository root.

**Download link:** https://drive.google.com/drive/folders/1VZ5WJbzCjdDs9v0zgrc9hlQas-ypGb2a?usp=drive_link

Download it and place its three folders (`checkpoints/`, `data/`, `real_data_experiment_output/`) directly in the repository root, so that the paths below exist.  The full download is about 6.4 GB, most of it `data/Ernie/`; Drive may split a large "Download all" into several zip files, in which case unzip them all into the repository root.

```
checkpoints/                                          # 10 mm source space, 50% training noise
├── multiembed_2src_10mm_noise50.ckpt                 # multi-embedding, 2 sources        (ablation.py)
├── topk_heatmap_2src_10mm_noise50.ckpt               # single-heatmap + top-k baseline   (ablation.py)
├── objectness_1to5src_10mm_noise50.ckpt              # multi-embedding + objectness, 1-5 sources
└── threshold_heatmap_1to5src_10mm_noise50.ckpt       # heatmap + threshold baseline, 1-5 sources
data/
├── Ernie/                                            # head model and simulation data
│   ├── ernie_eeg_simulations.mat                     # Electrode positions, dipole locations/orientations
│   ├── leadfield_patch5mmdistance.mat                # Leadfield matrix for 5 mm patch source space
│   ├── leadfield_patch10mmdistance.mat               # Leadfield matrix for 10 mm patch source space
│   ├── geodesic_distance_matrix_5mm_patches.mat
│   ├── geodesic_distance_matrix_10mm_patches.mat
│   ├── noise_vectors_5mm/                            # Pre-computed validation and test noise
│   │   ├── noise_vectors_val_50.npy
│   │   ├── noise_vectors_test_50.npy
│   │   └── ...
│   └── noise_vectors_10mm/
│       ├── noise_vectors_val_50.npy
│       ├── noise_vectors_test_50.npy
│       └── ...
├── motor/                                            # motor experiment (see below)
│   ├── sub_001.mat
│   └── sub_001_12SampleAroundpeak_490totalpeaks.mat
└── ds003645/sub-002/eeg/
    └── sub-002_task-FacePerception_run-1_electrodes.tsv   # electrode positions used by the motor experiment
real_data_experiment_output/                          # preprocessed face-experiment inputs
└── sub-00{2,3,4,5,6}_preprocessed.npz
```

The noise vector files are named by noise level (e.g. `_50` = 50% noise level).  If a file for a given noise level does not exist, it will be generated automatically on the first run and saved for reproducibility.  Each checkpoint is about 130 MB.  The leadfield is average-referenced (every source's column sums to zero across the 76 electrodes).

### Minimal download for the real-data experiments

To only run the real-EEG experiments on the pretrained checkpoints (see [Real-data experiments](#real-data-experiments)), you do not need the whole folder.  These files, about 0.7 GB in total, are sufficient (we ran every `--mode infer` command below with only these present):

- `data/Ernie/ernie_eeg_simulations.mat` (409 MB), `data/Ernie/leadfield_patch10mmdistance.mat` (1 MB) and `data/Ernie/geodesic_distance_matrix_10mm_patches.mat` (25 MB)
- `checkpoints/multiembed_2src_10mm_noise50.ckpt` and `checkpoints/objectness_1to5src_10mm_noise50.ckpt` (about 130 MB each)
- `real_data_experiment_output/sub-00{2,3,4,5,6}_preprocessed.npz` (for the face experiment)
- `data/motor/` and `data/ds003645/sub-002/eeg/…_electrodes.tsv` (for the motor experiment)

Training (`main.py`, `ablation.py`, `ablation_unknown_count.py`) needs the `data/Ernie/` files for the chosen source-space resolution, including its `noise_vectors_*` folder.

---

## Project structure

```
EEG-Transformer/
├── main.py                        # Single- and dual-source training/evaluation
├── ablation.py                    # n-source multi-embedding vs. single-heatmap top-k baseline
├── ablation_unknown_count.py      # Unknown source count: objectness vs. threshold-heatmap baseline
├── real_data_experiment.py        # Real EEG: face processing (N170), OpenNeuro ds003645
├── real_data_experiment_motor.py  # Real EEG: motor (button-press) response
├── clusterNetworks.py             # Network architectures
├── myDatasets.py                  # PyTorch Dataset classes
├── helpers.py                     # Data loading utilities
├── positional_encoding_3d.py      # 3D sinusoidal positional encoding
├── data/                          # Downloaded data (see above)
└── checkpoints/                   # Downloaded pretrained checkpoints (see above)
```

---

## Running experiments

All single- and dual-source experiments are launched through `main.py`.  The two required positional arguments are the **source space** (`5mm` or `10mm`) and the **noise level** (a fraction between 0 and 1).

```
python main.py <clusters> <noise_level> [options]
```

Training logs are written to `my_logs/` and can be monitored with TensorBoard:

```bash
tensorboard --logdir my_logs
```

The best checkpoint (by validation accuracy) is saved automatically and used for the final test evaluation.

The multi-source comparison, unknown-count, and real-data experiments use their own scripts; see [Multi-source comparison](#multi-source-comparison-ablationpy), [Unknown source count](#unknown-source-count-ablation_unknown_countpy), and [Real-data experiments](#real-data-experiments).

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

## Multi-source comparison (`ablation.py`)

The multi-embedding model is `clusterNetworks.TransformerEncoder` with `num_sources > 1`: every simultaneous source gets its own learned query embedding, and each embedding emits a softmax over the candidate source locations.  `ablation.py` generalizes the permutation-invariant loss of `main.py` from two sources to any `n` using optimal (Hungarian) bipartite matching, and compares it against the standard alternative in the EEG source-imaging literature: a **single heatmap** over all candidate locations read out with **top-k**.  Both models share the same electrode-tokenizing transformer backbone, so only the output representation and loss differ.

The top-k baseline is decoded with greedy non-maximum suppression (`--nms_radius`, geodesic mm, default 20; `0` gives plain top-k).  NMS is applied at evaluation time only; training is unchanged.

```bash
# Train and compare both models, 2 sources (use --tensorboard unless you have W&B set up)
python ablation.py 10mm 0.50 --num_sources 2 --method both --tensorboard

# Other source counts / one model only
python ablation.py 10mm 0.50 --num_sources 3 --method multiembed --tensorboard
python ablation.py 10mm 0.50 --num_sources 2 --method topk --tensorboard

# Re-evaluate a pretrained top-k checkpoint with and without NMS (no retraining)
python ablation.py 10mm 0.50 --num_sources 2 --eval_checkpoint checkpoints/topk_heatmap_2src_10mm_noise50.ckpt --nms_radius 20
python ablation.py 10mm 0.50 --num_sources 2 --eval_checkpoint checkpoints/topk_heatmap_2src_10mm_noise50.ckpt --nms_radius 0
```

Reported test metrics: `hit_rate` (fraction of true sources recovered), `exact_match` (all sources correct), and the optimally matched Euclidean and geodesic localization errors (`euclid_dist`, `geodist`, mm).  Logs and checkpoints go to `my_logs/ablation_cluster<res>_<n>src/noise_<level>/`.

Training logs to W&B by default (`--wandb_project`, `--wandb_entity` to choose where); pass `--tensorboard` to log locally instead.  A full 200-epoch run takes hours, so the pretrained checkpoints above are provided.

---

## Unknown source count (`ablation_unknown_count.py`)

Here the number of simultaneous sources is drawn at random from `[--min_sources, --max_sources]` (default 1–5) for every sample, and **neither model is told the count**.

- **Objectness model** (`--method objectness`): the multi-embedding model with an extra presence logit per query slot, trained against a padded "no object" class (DETR-style set prediction).  Slots whose presence probability exceeds `--presence_threshold` (default 0.5) form the predicted set.
- **Threshold-heatmap baseline** (`--method topk`): the single-heatmap model decoded by thresholding (`--prob_threshold`, default 0.5) with NMS (`--nms_radius`, default 20).

```bash
python ablation_unknown_count.py 10mm 0.50 --min_sources 1 --max_sources 5 --method both --tensorboard

# Re-evaluate pretrained checkpoints (no retraining)
python ablation_unknown_count.py 10mm 0.50 --eval_model objectness --eval_checkpoint checkpoints/objectness_1to5src_10mm_noise50.ckpt
python ablation_unknown_count.py 10mm 0.50 --eval_model threshold  --eval_checkpoint checkpoints/threshold_heatmap_1to5src_10mm_noise50.ckpt
```

Because predicted and true sets can differ in size, the reported metrics are precision, recall (`hit_rate`), F1, `exact_match`, count error (`count_mae`, `count_acc`), and localization error.  For localization error **use `chamfer_p2t_euclid` / `chamfer_p2t_geo`** (mean distance from every *predicted* location to its nearest true source).  The matched `euclid_dist` / `geodist` only score `min(#predicted, #true)` pairs, so a method that over-predicts is not penalized for its extra predictions; the `chamfer_t2p_*` direction (true to nearest predicted) has the same blind spot.

---

## Real-data experiments

Both experiments apply a trained Ernie checkpoint to real EEG **without retraining or fine-tuning**.  The real electrode montage (70 channels) is interpolated onto Ernie's 76 electrodes (inverse-distance weighting between unit-sphere-normalized layouts), and the signal's overall amplitude is rescaled to the typical norm of the training distribution, since the network has no input-normalization layer and real EEG (volts) is ~10⁷× smaller than the leadfield's arbitrary units.  Only the overall gain is matched, not the spatial pattern.  Predicted locations are in Ernie's coordinate frame.

Both scripts take `--model multiembed` (fixed `--num_sources`, default 2) or `--model objectness` (variable count, reports a presence probability per slot).

### Face processing (N170), OpenNeuro ds003645

Data: [ds003645](https://openneuro.org/datasets/ds003645) (CC0; Wakeman & Henson, 2015, *Sci. Data* 2:150001; DOI `10.18112/openneuro.ds003645.v2.0.2`), subjects `sub-002`–`sub-006`.  The model input is each subject's grand-average, baseline-corrected N170 topography (mean over 130–200 ms minus the −200 to 0 ms baseline, averaged over about 590 face trials).  No noise is added: the noise in the input is the real background EEG left after averaging.

**Quick path (no raw data needed):** the five `sub-00X_preprocessed.npz` files are included in the download (`real_data_experiment_output/`), so inference can be run directly:

```bash
python real_data_experiment.py --mode infer --model multiembed --checkpoint checkpoints/multiembed_2src_10mm_noise50.ckpt
python real_data_experiment.py --mode infer --model objectness --checkpoint checkpoints/objectness_1to5src_10mm_noise50.ckpt
```

**From the raw EEG (about 4.8 GB, needs `mne`):** download the five subjects from OpenNeuro (for example with the AWS CLI as below; `openneuro-py` or DataLad also work), then preprocess:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds003645 data/ds003645 --exclude "*" --include "sub-00[2-6]/*"
python real_data_experiment.py --mode preprocess --subjects sub-002 sub-003 sub-004 sub-005 sub-006
```

For each subject the script prints both predicted slots with their confidence, position (mm), and the distance to the nearest right occipital face area (OFA) coordinate, (45, −80, −12) from Pitcher et al. (2009) or (46, −75, −3) from Bona et al. (2015) (MNI, ~10 mm apart).  The paper's face table reports, for each subject, the slot closest to those coordinates.  Expected values with the checkpoint above:

| Subject | Position (mm) | Distance to nearest OFA coordinate |
|---|---|---|
| sub-002 | (33.7, −56.8, 37.2) | 45.8 mm |
| sub-003 | (34.1, −83.3, 26.2) | 32.6 mm |
| sub-004 | (30.7, −55.2, 8.4) | 27.4 mm |
| sub-005 | (29.6, −13.1, 0.4) | 64.1 mm |
| sub-006 | (34.1, −83.3, 26.2) | 32.6 mm |

**Note on the EEG reference.**  ds003645 was recorded against a nose reference, and the script does not re-reference it, whereas the Ernie leadfield is average-referenced.  This matches the paper.  In a sensitivity check, subtracting the channel mean from the interpolated signal left the predicted regions unchanged for 3 of 5 subjects and changed them for the other 2.  To run with a strict common-average reference, subtract the channel mean from `ernie_signal` before rescaling.

### Motor (button-press) response

Data (`data/motor/`, provided by a collaborator): subject 001's response to a right-hand index-finger button press, from the same underlying recording as ds003645 but response-locked (70 channels, 1100 Hz, 1–40 Hz band-pass, average reference, baseline −300 to −200 ms).  The input is the mean over 490 single trials in the 20–30 ms post-press window.  The reference location is the collaborator's own eLORETA estimate on that average, (−22.0, 4.0, 49.5) mm, left hemisphere and dorsal/central (consistent with contralateral left primary motor cortex).  That estimate is another method's output, not a verified ground truth, and is expressed in the subject's own head-model coordinates, so the distances the script prints are rough plausibility proxies rather than validated errors.

Electrode positions are borrowed from `sub-002`'s `electrodes.tsv` (the same 70-channel cap layout; subject 001's own digitization is not available).  That one small file is included in the download, so the full ds003645 download is **not** needed for this experiment.

```bash
python real_data_experiment_motor.py --model multiembed --checkpoint checkpoints/multiembed_2src_10mm_noise50.ckpt
python real_data_experiment_motor.py --model objectness --checkpoint checkpoints/objectness_1to5src_10mm_noise50.ckpt
```

Expected output: the 2-source model puts slot 0 at region 1060, (−7.6, 2.5, 71.9) mm, 26.7 mm from the reference (slot 1 lands far away, at (5.3, −44.7, −32.5) mm, 99.2 mm); the objectness model activates 3 of 5 slots, at regions 1060, 738 and 1060 again, all in the left hemisphere.  The reported noise level is the trial-to-trial standard deviation across the 490 trials, for context only; it is not an input to the model.

---

## Acknowledgments and references

- Wakeman, D. G. & Henson, R. N. (2015). A multi-subject, multi-modal human neuroimaging dataset. *Scientific Data* 2, 150001.
- Pitcher, D., Charles, L., Devlin, J. T., Walsh, V. & Duchaine, B. (2009). Triple dissociation of faces, bodies, and objects in extrastriate cortex. *Current Biology* 19(4), 319–324.
- Bona, S., Cattaneo, Z. & Silvanto, J. (2015). The causal role of the occipital face area (OFA) and lateral occipital (LO) cortex in symmetry perception. *Journal of Neuroscience* 35(2), 731–738.

The motor-experiment data were provided by a collaborator; see the paper's acknowledgments.
