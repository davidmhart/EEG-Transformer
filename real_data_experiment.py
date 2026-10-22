"""
real_data_experiment.py -- does the Ernie-trained multi-embedding model generalize to real EEG,
localizing to plausible face-processing generators?

Uses OpenNeuro ds003645 (BIDS conversion of Wakeman & Henson's multimodal face-processing
dataset, ds000117), EEG channels only (see data/ds003645/, downloaded separately). The model
input is the grand-average N170 topography: for each face trial, the mean over 130-200 ms
minus the mean over the -200 to 0 ms pre-stimulus baseline, per channel, averaged across
trials. No noise is added to this signal; the noise in the input is the real background EEG
and sensor noise left after averaging ~590 trials. The pre-stimulus baseline is used for
baseline correction and to compute a per-channel single-trial noise level (the SD across
trials of the baseline-window mean, saved as `trial_baseline_std_real`), which is descriptive
only and is not an input to the model. The real montage (Easycap 70-site, Neuromag/BIDS
convention) does not match Ernie's 76-electrode layout, so real data is spatially interpolated
onto Ernie's electrode positions before being fed to the existing checkpoint -- no retraining.

REFERENCE (read this before comparing to the paper's numbers)
--------------------------------------------------------------
The Ernie leadfield is average-referenced (each source's column sums to zero across the 76
electrodes). ds003645 was recorded against a nose reference (EEGLAB header: ref='common'), and
this script does NOT re-reference it: the interpolated signal is fed to the model as recorded,
so it carries a common-mode component that a common-average-referenced signal would not. This
matches the results reported in the paper. In a sensitivity check, subtracting the channel mean
from the interpolated 76-channel signal left the predicted regions unchanged for 3 of 5
subjects and changed them for the other 2; users who want a strictly common-average-referenced
input can apply `signal - signal.mean()` to `ernie_signal` before rescaling. The motor data
(real_data_experiment_motor.py) are already average-referenced.

MONTAGE / UNITS NOTES (worth re-deriving if this is ever rerun on a different BIDS dataset,
rather than trusted blindly)
--------------------------------------------------------------------------------------------
- ds003645's electrodes.tsv values are labeled "mm" in coordsystem.json but are actually
  ~10x too small to be mm on a real head (electrode vector norms ~7-13, vs. a real ~70-130mm
  head radius) and match the coordsystem.json's own (already-mm) fiducial coordinates almost
  exactly when multiplied by 10 (e.g. the most anterior electrode's y-component times 10 lands
  within ~2% of the Nasion y-coordinate). So electrodes.tsv is treated here as centimeters,
  scaled x10 to mm, despite the mislabeled unit in the file.
- Ernie's electrode_locs (helpers.load_electrode_locs) and the real montage are in different
  coordinate frames (different origins -- Neuromag's is the LPA/RPA midpoint; Ernie's appears
  to be a head centroid -- and probably different absolute scale/head size), but both appear to
  use the same axis convention (x = left-right, y = posterior->anterior, z = inferior->superior;
  confirmed qualitatively -- Ernie's most anterior electrodes have the largest +y, matching the
  real montage's nasion-directed +y). Given that, both montages are centered on their own
  centroid and radially normalized to a unit sphere (removing origin and scale differences)
  before interpolation, so only *angular* electrode position is used -- exactly what an
  inverse-distance-weighted spherical interpolation needs to be meaningful.
- No electrode-name correspondence exists between the two montages (Ernie's are unlabeled,
  the real montage's are generic EEG001-074, not 10-20 names), so this interpolation is
  necessarily geometric/approximate, not identity-verified per channel. Flag this as a real
  limitation, not a hidden detail.

TWO MODES
---------
--mode preprocess needs mne (reads the raw EEGLAB files; does not need torch).
--mode infer needs torch/pytorch-lightning/hdf5storage (does not need mne).
They hand off through saved .npz files in real_data_experiment_output/, so they can run in one
environment that has everything installed, or in two separate ones. The preprocessed .npz files
can also be downloaded directly (see the README), which skips the raw-data download and mne.

MODEL CHOICE: fixed 2-source vs. variable/unknown-count ("objectness")
------------------------------------------------------------------------
--model multiembed (default) forces exactly --num_sources output slots (LitMultiEmbedNSource,
from ablation.py). 2 sources is a defensible prior for the N170 window -- the literature
consistently reports *bilateral* occipitotemporal generators (right-lateralized in amplitude,
not exclusive to the right hemisphere; e.g. Deffke et al., 2007, combined EEG-fMRI localization
of N170 to bilateral fusiform sources; Rossion & Jacques, 2008 review) -- but it is the
configuration the checkpoint was trained with (10mm/noise=0.50/2-source), not the result of a
source-count analysis run on this dataset. A fixed-slot model has no way to say
"I only see one source here": if a subject's real response is actually single-source-dominant,
the second slot still has to output *something*, which can produce a generic/degenerate
location rather than a genuine second generator -- a plausible explanation if the same
"filler" location keeps recurring across subjects.

--model objectness instead runs LitObjectnessNSource (from ablation_unknown_count.py, trained
with a random 1-5 source count and an explicit per-slot presence/objectness head -- see that
module's docstring). It doesn't assume a source count: each of --max_sources slots reports its
own presence probability, and only slots clearing --presence_threshold are treated as real
predictions. This lets the model itself say "1" or "2" (or more) per subject instead of us
imposing 2, and is a direct check on whether repeated, generic-looking predictions across
subjects are a fixed-cardinality artifact.

Usage:
    # needs mne and the raw ds003645 EEG files (skip if you downloaded the preprocessed .npz):
    python real_data_experiment.py --mode preprocess --subjects sub-002 sub-003 sub-004 sub-005 sub-006

    # needs torch / pytorch-lightning / hdf5storage:
    python real_data_experiment.py --mode infer --model multiembed \\
        --checkpoint checkpoints/multiembed_2src_10mm_noise50.ckpt
    python real_data_experiment.py --mode infer --model objectness \\
        --checkpoint checkpoints/objectness_1to5src_10mm_noise50.ckpt
"""

import argparse
import glob
import json
import os

import numpy as np


DATA_ROOT = "data/ds003645"
OUTPUT_DIR = "real_data_experiment_output"
REAL_UNIT_TO_MM = 10.0  # see module docstring
# Right occipital face area (OFA) functional-localizer peaks (MNI, mm), used only to report a
# distance for each predicted slot: Pitcher et al. 2009 (Curr. Biol. 19:319-324) and Bona et
# al. 2015 (J. Neurosci. 35:731-738); the two are ~10 mm apart. The comparison assumes Ernie's
# coordinate frame is approximately MNI-aligned, which has not been verified.
OFA_TARGETS_MM = {"Pitcher09": (45.0, -80.0, -12.0), "Bona15": (46.0, -75.0, -3.0)}
BASELINE_WINDOW = (-0.2, 0.0)   # seconds relative to stimulus onset
N170_WINDOW = (0.130, 0.200)    # seconds relative to stimulus onset -- classic N170/M170 window
EPOCH_WINDOW = (-0.2, 0.4)      # seconds, generous enough to cover both windows above


def _nearest_ofa_distance(pos_mm):
    """(name, distance in mm) of the nearest right-OFA literature coordinate."""
    d = {name: float(np.linalg.norm(np.asarray(pos_mm, dtype=float) - np.array(t)))
         for name, t in OFA_TARGETS_MM.items()}
    best = min(d, key=d.get)
    return best, d[best]


def load_real_montage(subject, run=1):
    """Real electrode names/positions (mm, in the real dataset's own head-coordinate frame),
    restricted to genuine scalp EEG sites (drops HEOG/VEOG/ECG/MISC/trigger channels)."""
    import pandas as pd
    elec_path = f"{DATA_ROOT}/{subject}/eeg/{subject}_task-FacePerception_run-{run}_electrodes.tsv"
    df = pd.read_csv(elec_path, sep="\t")
    df = df.dropna(subset=["x", "y", "z"])  # drops the n/a (non-scalp) rows
    names = df["name"].tolist()
    pos_mm = df[["x", "y", "z"]].to_numpy(dtype=np.float64) * REAL_UNIT_TO_MM
    return names, pos_mm


def _sphere_normalize(pos):
    """Center on centroid and project to the unit sphere (unit vectors from the centroid) --
    removes origin/scale differences between two montages, leaving only angular position."""
    center = pos.mean(axis=0)
    centered = pos - center
    radii = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / radii


def build_interpolation_matrix(real_pos_mm, ernie_pos_mm, k=5):
    """(n_ernie, n_real) inverse-distance weight matrix on the unit sphere: row i gives the
    weights used to synthesize Ernie electrode i's signal from the k nearest real electrodes
    (by angular position). Same style as ConvDipNetwork's electrode interpolator elsewhere in
    this project, applied here to remap between two *electrode* layouts rather than to a 2D
    grid image.
    """
    real_unit = _sphere_normalize(real_pos_mm)
    ernie_unit = _sphere_normalize(ernie_pos_mm)

    # chordal distance on the unit sphere (monotonic in angular distance -- fine for ranking/
    # weighting nearest neighbors, no need for the exact great-circle angle)
    dist = np.linalg.norm(ernie_unit[:, None, :] - real_unit[None, :, :], axis=-1)  # (n_ernie, n_real)
    idx = np.argsort(dist, axis=1)[:, :k]
    nearest_d = np.take_along_axis(dist, idx, axis=1)
    weights = 1.0 / (nearest_d + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    n_ernie, n_real = ernie_pos_mm.shape[0], real_pos_mm.shape[0]
    interp = np.zeros((n_ernie, n_real), dtype=np.float64)
    for i in range(n_ernie):
        interp[i, idx[i]] = weights[i]
    return interp


def process_subject(subject, ernie_pos_mm):
    import mne
    import pandas as pd

    real_names, real_pos_mm = load_real_montage(subject)
    interp = build_interpolation_matrix(real_pos_mm, ernie_pos_mm)
    print(f"[{subject}] real montage: {len(real_names)} channels; interpolation matrix "
         f"{interp.shape} built")

    run_paths = sorted(glob.glob(f"{DATA_ROOT}/{subject}/eeg/{subject}_task-FacePerception_run-*_eeg.set"))
    signal_epochs = []   # (n_trials_total, n_real_channels) -- mean amplitude in the N170 window
    baseline_epochs = [] # same, for the pre-stimulus baseline window

    for set_path in run_paths:
        run = os.path.basename(set_path).split("run-")[1].split("_")[0]
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        raw.pick(real_names)  # scalp EEG only, in the same order as real_names/interp's columns
        sfreq = raw.info["sfreq"]
        data = raw.get_data()  # (n_channels, n_times), volts

        events_path = f"{DATA_ROOT}/{subject}/{subject}_task-FacePerception_run-{run}_events.tsv"
        events = pd.read_csv(events_path, sep="\t")
        face_events = events[
            events["event_type"].isin(["show_face", "show_face_initial"])
            & events["face_type"].isin(["famous_face", "unfamiliar_face"])
        ]

        n_pre = int(round(EPOCH_WINDOW[0] * sfreq))
        n_post = int(round(EPOCH_WINDOW[1] * sfreq))
        for onset_sec in face_events["onset"]:
            onset_sample = int(round(onset_sec * sfreq))
            start, stop = onset_sample + n_pre, onset_sample + n_post
            if start < 0 or stop > data.shape[1]:
                continue
            epoch = data[:, start:stop]
            t = np.arange(n_pre, n_post) / sfreq

            base_mask = (t >= BASELINE_WINDOW[0]) & (t < BASELINE_WINDOW[1])
            sig_mask = (t >= N170_WINDOW[0]) & (t < N170_WINDOW[1])
            baseline_epochs.append(epoch[:, base_mask].mean(axis=1))
            signal_epochs.append(epoch[:, sig_mask].mean(axis=1) - epoch[:, base_mask].mean(axis=1))

        print(f"  run-{run}: {len(face_events)} face events, "
             f"{sum(1 for _ in face_events['onset'])} attempted")

    signal_epochs = np.array(signal_epochs)      # (n_trials, n_real_channels)
    baseline_epochs = np.array(baseline_epochs)  # (n_trials, n_real_channels)
    print(f"[{subject}] {signal_epochs.shape[0]} usable face trials total across all runs")

    grand_avg_signal = signal_epochs.mean(axis=0)      # (n_real_channels,) baseline-corrected N170 topography
    grand_avg_baseline = baseline_epochs.mean(axis=0)  # (n_real_channels,) mean pre-stim level (near 0 in volts, expected)
    trial_baseline_std = baseline_epochs.std(axis=0)   # (n_real_channels,) real single-trial noise level

    ernie_signal = interp @ grand_avg_signal            # (n_ernie_channels,)
    ernie_baseline_std = interp @ trial_baseline_std    # (n_ernie_channels,) interpolated real noise level

    return {
        "subject": subject,
        "n_trials": int(signal_epochs.shape[0]),
        "real_names": real_names,
        "real_pos_mm": real_pos_mm,
        "grand_avg_signal_real": grand_avg_signal,
        "trial_baseline_std_real": trial_baseline_std,
        "ernie_signal": ernie_signal,
        "ernie_baseline_std": ernie_baseline_std,
        "interp_matrix": interp,
    }


def run_preprocess(subjects):
    import scipy.io

    # Loaded directly via scipy rather than helpers.load_electrode_locs (which requires
    # hdf5storage) so that this mode only needs mne/scipy/pandas, not the training stack.
    ernie_pos_mm = scipy.io.loadmat("data/Ernie/ernie_eeg_simulations.mat",
                                    variable_names=["electrodes"])["electrodes"].astype(np.float64)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for subject in subjects:
        result = process_subject(subject, ernie_pos_mm)
        out_path = os.path.join(OUTPUT_DIR, f"{subject}_preprocessed.npz")
        np.savez(out_path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                subject=result["subject"], n_trials=result["n_trials"])
        print(f"Wrote {out_path}")


def _seeded_mean_norm(dataset, n_samples, seed):
    """Mean L2 norm of the first `n_samples` dataset items, with numpy AND torch RNGs seeded
    (sources are drawn with numpy, additive noise with torch) so the calibration scale -- and
    therefore any borderline prediction -- is identical from run to run. Prior RNG state is
    restored afterwards."""
    import torch

    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        norms = np.array([np.linalg.norm(dataset[i][0].numpy()) for i in range(n_samples)])
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    return float(norms.mean())


def _training_scale_reference(clusters, noise_level, num_sources, n_samples=2000, seed=0):
    """
    Mean L2 norm of a representative training-distribution signal (leadfield-units summed
    sources + noise, same convention the checkpoint was actually trained on) -- the reference
    scale real data needs to be rescaled to before this network's input makes sense.

    Why this is necessary: the network has no input normalization layer, so it's scale-
    sensitive by construction. Training signals are in the leadfield's own arbitrary numeric
    units (mean |value| ~20-30 for this project's 10mm leadfield); real EEG is in volts
    (~1e-6). Feeding raw real data straight in would present the network with an input ~1e7x
    smaller than anything it was trained on -- not "a hard example," a different regime the
    weights were never calibrated for. There's no first-principles "correct" scale factor
    connecting an arbitrary synthetic dipole-strength unit to physical EEG amplitude; matching
    the real signal's overall gain (not its cross-channel pattern, which is what actually
    carries localization information) to the training distribution's typical norm is the
    standard, defensible thing to do in this situation, and is flagged here rather than buried.
    """
    from helpers import load_data, load_dist_table
    from ablation import NSourceDataset

    leadfield, _ = load_data(clusters)
    dist_table = load_dist_table(clusters)
    num_dipoles = leadfield.shape[1]
    ds = NSourceDataset(num_dipoles, leadfield, n_samples, num_sources=num_sources,
                        dist_table=dist_table, noise_level=noise_level)
    return _seeded_mean_norm(ds, n_samples, seed)


def _training_scale_reference_variable(clusters, noise_level, min_sources, max_sources, n_samples=2000, seed=0):
    """
    Same idea as `_training_scale_reference` above, but for the objectness model's training
    distribution: VariableNSourceDataset (random n in [min_sources, max_sources] per sample),
    not a fixed-n NSourceDataset. Kept separate rather than parameterizing one function over
    both, since the two datasets have different constructors/semantics (equal-strength-only,
    no `num_sources` argument) and conflating them would obscure which convention is in play.
    """
    from helpers import load_data, load_dist_table
    from ablation_unknown_count import VariableNSourceDataset

    leadfield, _ = load_data(clusters)
    dist_table = load_dist_table(clusters)
    num_dipoles = leadfield.shape[1]
    ds = VariableNSourceDataset(num_dipoles, leadfield, n_samples, min_sources=min_sources,
                                max_sources=max_sources, dist_table=dist_table, noise_level=noise_level)
    return _seeded_mean_norm(ds, n_samples, seed)


def run_infer_objectness(checkpoint, clusters="10mm", noise_level=0.50, min_sources=1, max_sources=5,
                         presence_threshold=0.5, pe="none", dropout_num=0):
    """
    Variable/unknown-count counterpart of run_infer: instead of forcing exactly num_sources
    output slots, runs the DETR-style objectness model (LitObjectnessNSource) and reports every
    slot's own presence probability alongside its argmax location -- the model decides how many
    of its max_sources slots represent real sources, rather than us imposing a count. See the
    module docstring's "MODEL CHOICE" section for why this matters here.
    """
    import torch
    from helpers import load_data, load_locs, load_dist_table, load_electrode_locs
    from ablation_unknown_count import LitObjectnessNSource

    leadfield, _ = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    ref_norm = _training_scale_reference_variable(clusters, noise_level, min_sources, max_sources)
    print(f"Training-distribution reference ||signal||_2 = {ref_norm:.2f} (leadfield units)")

    model = LitObjectnessNSource.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        max_sources=max_sources, cluster_locs=cluster_locs, dist_table=dist_table, pe=pe,
        dropout_num=dropout_num, presence_threshold=presence_threshold,
    )
    model.eval()

    results = {}
    for npz_path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_preprocessed.npz"))):
        d = np.load(npz_path, allow_pickle=True)
        subject = str(d["subject"])
        raw_signal = d["ernie_signal"]  # (76,), volts, baseline-corrected N170 topography

        raw_norm = np.linalg.norm(raw_signal)
        scale = ref_norm / raw_norm
        scaled_signal = raw_signal * scale
        print(f"[{subject}] raw ||signal||_2={raw_norm:.3e}V, rescaled by {scale:.3e} to match "
             f"training distribution (new norm={np.linalg.norm(scaled_signal):.2f})")

        signal_t = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            loc_logits, presence_logits = model.model(signal_t)         # (1, max_sources, C), (1, max_sources)
            loc_probs = torch.softmax(loc_logits, dim=-1).squeeze(0).numpy()
            presence_probs = torch.sigmoid(presence_logits).squeeze(0).numpy()
        preds = loc_probs.argmax(axis=1)
        pred_pos = cluster_locs[preds]
        active = presence_probs > presence_threshold

        print(f"[{subject}] {int(active.sum())}/{max_sources} slot(s) cleared "
             f"presence_threshold={presence_threshold}:")
        for slot in range(max_sources):
            flag = "ACTIVE" if active[slot] else "  --  "
            target, dist = _nearest_ofa_distance(pred_pos[slot])
            print(f"  slot{slot} [{flag}]: presence_p={presence_probs[slot]:.4f}  "
                 f"region={preds[slot]}  loc_p={loc_probs[slot, preds[slot]]:.4f}  "
                 f"pos_mm={pred_pos[slot]}  | nearest R-OFA target: {target}, {dist:.1f} mm")

        results[subject] = {"preds": preds, "loc_probs": loc_probs, "presence_probs": presence_probs,
                            "active": active, "pred_pos_mm": pred_pos, "scale_factor": scale,
                            "n_trials": int(d["n_trials"])}

    out_path = os.path.join(OUTPUT_DIR, "inference_results_objectness.npz")
    np.savez(out_path, **{f"{sub}_{k}": v for sub, r in results.items() for k, v in r.items()})
    print(f"\nWrote {out_path}")
    return results


def run_infer(checkpoint, clusters="10mm", noise_level=0.50, num_sources=2, pe="none", dropout_num=0):
    import torch
    from helpers import load_data, load_locs, load_dist_table, load_electrode_locs
    from ablation import LitMultiEmbedNSource

    leadfield, _ = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    ref_norm = _training_scale_reference(clusters, noise_level, num_sources)
    print(f"Training-distribution reference ||signal||_2 = {ref_norm:.2f} (leadfield units)")

    model = LitMultiEmbedNSource.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        num_sources=num_sources, cluster_locs=cluster_locs, dist_table=dist_table, pe=pe,
        dropout_num=dropout_num,
    )
    model.eval()

    results = {}
    for npz_path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_preprocessed.npz"))):
        d = np.load(npz_path, allow_pickle=True)
        subject = str(d["subject"])
        raw_signal = d["ernie_signal"]  # (76,), volts, baseline-corrected N170 topography

        raw_norm = np.linalg.norm(raw_signal)
        scale = ref_norm / raw_norm
        scaled_signal = raw_signal * scale
        print(f"[{subject}] raw ||signal||_2={raw_norm:.3e}V, rescaled by {scale:.3e} to match "
             f"training distribution (new norm={np.linalg.norm(scaled_signal):.2f})")

        signal_t = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model.model(signal_t)               # (1, num_sources, out_channels)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        preds = probs.argmax(axis=1)
        pred_pos = cluster_locs[preds]

        print(f"[{subject}] predicted {num_sources} source(s):")
        for slot in range(num_sources):
            target, dist = _nearest_ofa_distance(pred_pos[slot])
            print(f"  slot{slot}: region={preds[slot]}  p={probs[slot, preds[slot]]:.4f}  "
                 f"pos_mm={pred_pos[slot]}  | nearest R-OFA target: {target}, {dist:.1f} mm")

        results[subject] = {"preds": preds, "probs": probs, "pred_pos_mm": pred_pos,
                            "scale_factor": scale, "n_trials": int(d["n_trials"])}

    out_path = os.path.join(OUTPUT_DIR, "inference_results.npz")
    np.savez(out_path, **{f"{sub}_{k}": v for sub, r in results.items() for k, v in r.items()})
    print(f"\nWrote {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real EEG (ds003645) -> Ernie-montage interpolation "
                                                 "-> multi-embedding source localization")
    parser.add_argument("--mode", type=str, required=True, choices=["preprocess", "infer"])
    parser.add_argument("--subjects", type=str, nargs="+", default=["sub-002", "sub-003"])
    parser.add_argument("--model", type=str, default="multiembed", choices=["multiembed", "objectness"],
                        help="multiembed: fixed --num_sources slots (ablation.LitMultiEmbedNSource). "
                             "objectness: variable/unknown count with a per-slot presence head "
                             "(ablation_unknown_count.LitObjectnessNSource). See module docstring.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint matching --model (required for --mode infer)")
    parser.add_argument("--clusters", type=str, default="10mm")
    parser.add_argument("--noise_level", type=float, default=0.50)
    parser.add_argument("--num_sources", type=int, default=2, help="--model multiembed only")
    parser.add_argument("--min_sources", type=int, default=1, help="--model objectness only")
    parser.add_argument("--max_sources", type=int, default=5, help="--model objectness only")
    parser.add_argument("--presence_threshold", type=float, default=0.5, help="--model objectness only")
    parser.add_argument("--pe", type=str, default="none")
    parser.add_argument("--dropout_num", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocess(args.subjects)
    else:
        if not args.checkpoint:
            raise SystemExit("--mode infer requires --checkpoint")
        if args.model == "objectness":
            run_infer_objectness(args.checkpoint, args.clusters, args.noise_level, args.min_sources,
                                 args.max_sources, args.presence_threshold, args.pe, args.dropout_num)
        else:
            run_infer(args.checkpoint, args.clusters, args.noise_level, args.num_sources,
                     args.pe, args.dropout_num)
