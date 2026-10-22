"""
real_data_experiment_motor.py -- apply a trained checkpoint to real motor (button-press) EEG.

Companion to real_data_experiment.py (face-processing N170): same montage interpolation and
amplitude calibration, different data. No retraining or fine-tuning.

DATA / PROVENANCE
-----------------
The files in data/motor/ were provided by a collaborator. They are derived from the same
underlying recording as ds003645 (OpenNeuro; Wakeman & Henson 2015, ds000117), but
response-locked to the RIGHT-hand index-finger button press instead of stimulus onset.
Preprocessing, as provided: 70 EEG channels at 1100 Hz, epochs from -500 to +500 ms around the
press, 1-40 Hz zero-phase 4th-order Butterworth band-pass, average reference, baseline
correction using -300 to -200 ms, trials with >150 uV peak-to-peak amplitude rejected.
For subject 001:
  - sub_001.mat: a subject-specific 70 x 8196 BEM leadfield, 8196 cortical `source_pos`
    (meters), and `eeg_labels` (channel names) -- that subject's own forward model, not Ernie's.
  - sub_001_12SampleAroundpeak_490totalpeaks.mat: `eeg_peak_trials`, shape (70, 12, 490) --
    single-trial sensor data (volts): 12 samples (~11 ms at 1100 Hz) spanning the 20-30 ms
    post-press window used for source localization, for all 490 right-press trials.
Only subject 001 is used.

REFERENCE LOCATION, AND A CAVEAT
---------------------------------
The collaborator's eLORETA estimate on the 490-trial average is source index 2388 (MATLAB,
1-indexed) of sub_001's `source_pos`; converted from meters to mm that is (-22.02, 3.98, 49.53).
That is left hemisphere (this project's convention: +x = right, -x = left) and
dorsal/central -- consistent with the classic contralateral (left) primary motor cortex
hand-area response to a right-hand button press.

This is another algorithm's estimate, not an independently verified ground truth, and it is
expressed in that subject's own BEM coordinate frame, which has NOT been verified to align with
the Ernie frame the checkpoints were trained in. Any Euclidean distance this script prints
between a prediction and that coordinate is therefore a rough plausibility proxy, not a
validated localization error; the qualitative check (hemisphere, dorsal/central position) is the
comparison to trust.

NOISE
-----
Neither noise quantity here is an input to the model, which takes only the interpolated,
amplitude-calibrated grand-average vector. The epochs in these files were already
baseline-corrected and no separate pre-press segment is included, so the single-trial noise
level is characterized as the trial-to-trial standard deviation across the 490 trials in the
response window. That is real recorded noise, but a weaker isolation than the pre-stimulus
baseline used for the face data in real_data_experiment.py, since it partially conflates
response-locked variability with sensor noise. It is reported for SNR context only.

ELECTRODE POSITIONS
-------------------
The channel labels ('EEG001'..'EEG060', 'EEG065'..'EEG074') match ds003645's electrodes.tsv
naming exactly (same recording system), but this subject's own digitized positions are not
included. The script borrows another ds003645 subject's electrodes.tsv (default: sub-002) as a
geometric stand-in; the same 70-channel cap layout is used across subjects, but this is an
approximation, not subject 001's own digitization. Only that one small file
(data/ds003645/sub-002/eeg/sub-002_task-FacePerception_run-1_electrodes.tsv) is needed, not
the full dataset.

Reuses real_data_experiment.py's montage-interpolation and scale-calibration functions.

Usage:
    python real_data_experiment_motor.py --model multiembed \
        --checkpoint checkpoints/multiembed_2src_10mm_noise50.ckpt
    python real_data_experiment_motor.py --model objectness \
        --checkpoint checkpoints/objectness_1to5src_10mm_noise50.ckpt
"""

import argparse

import numpy as np
import scipy.io

from real_data_experiment import (build_interpolation_matrix, _training_scale_reference,
                                  _training_scale_reference_variable)


MOTOR_DATA_ROOT = "data/motor"
PROXY_MONTAGE_SUBJECT = "sub-002"   # stand-in for sub-001's digitized positions -- see docstring
PROXY_MONTAGE_ROOT = "data/ds003645"
REAL_UNIT_TO_MM = 10.0              # same mislabeled-cm-as-mm fix as real_data_experiment.py
GT_SOURCE_INDEX_1BASED = 2388        # reported eLORETA source index (MATLAB, 1-indexed)


def load_motor_trial_data(subject="sub_001"):
    """Per-trial sensor data + channel labels for the given subject."""
    d = scipy.io.loadmat(f"{MOTOR_DATA_ROOT}/{subject}_12SampleAroundpeak_490totalpeaks.mat")
    trials = d["eeg_peak_trials"]  # (n_channels, n_samples, n_trials), volts
    labels = [str(x[0]) for x in scipy.io.loadmat(f"{MOTOR_DATA_ROOT}/{subject}.mat")["eeg_labels"].flatten()]
    return trials, labels


def load_motor_ground_truth(subject="sub_001", source_index_1based=GT_SOURCE_INDEX_1BASED):
    """The subject-specific leadfield/source_pos, and the mm coordinate for the reported
    eLORETA source index. See module docstring for the coordinate-frame caveat."""
    d = scipy.io.loadmat(f"{MOTOR_DATA_ROOT}/{subject}.mat")
    source_pos_mm = d["source_pos"] * 1000.0  # meters -> mm
    coord_mm = source_pos_mm[source_index_1based - 1]
    return {"leadfield": d["leadfield"], "source_pos_mm": source_pos_mm, "coord_mm": coord_mm,
            "source_index_1based": source_index_1based}


def load_proxy_real_montage(subject=PROXY_MONTAGE_SUBJECT):
    """Stand-in real electrode positions (mm) for subject 001 -- see module docstring's
    ELECTRODE POSITIONS section for why this is an approximation, not subject-001-exact."""
    import pandas as pd
    elec_path = f"{PROXY_MONTAGE_ROOT}/{subject}/eeg/{subject}_task-FacePerception_run-1_electrodes.tsv"
    df = pd.read_csv(elec_path, sep="\t").dropna(subset=["x", "y", "z"])
    names = df["name"].tolist()
    pos_mm = df[["x", "y", "z"]].to_numpy(dtype=np.float64) * REAL_UNIT_TO_MM
    return names, pos_mm


def _prepare_real_input(clusters, subject, proxy_montage_subject):
    """Shared montage-interpolation step for both model choices below: the grand-average
    signal + real single-trial noise std, both interpolated onto Ernie's electrode layout, plus
    the reference location and Ernie's own electrode positions (needed again by the caller for model
    construction)."""
    from helpers import load_electrode_locs

    trials, motor_labels = load_motor_trial_data(subject)
    gt = load_motor_ground_truth(subject)
    real_names, real_pos_mm = load_proxy_real_montage(proxy_montage_subject)

    if motor_labels != real_names:
        raise ValueError(
            f"Channel name/order mismatch between {subject}'s eeg_labels and "
            f"{proxy_montage_subject}'s electrodes.tsv -- interpolation would silently apply "
            f"the wrong positions to the wrong channels. Check both files before proceeding."
        )

    # Grand average over all 490 trials and all 12 peri-peak samples -- exactly the input the reference
    # eLORETA analysis used ("mean sensor activity from 20-30ms after the button press").
    grand_avg_signal = trials.mean(axis=(1, 2))    # (n_channels,) volts
    trial_std = trials.std(axis=2).mean(axis=1)    # (n_channels,) real single-trial noise -- see
                                                    # module docstring's REAL NOISE section

    electrode_locs = load_electrode_locs()
    interp = build_interpolation_matrix(real_pos_mm, electrode_locs)
    ernie_signal = interp @ grand_avg_signal
    ernie_noise_std = interp @ trial_std

    return ernie_signal, ernie_noise_std, gt, electrode_locs


def _report_prediction(subject, preds, probs_or_presence, pred_pos, gt, label="p"):
    print(f"\n[{subject}] predicted {len(preds)} source(s):")
    for slot, (region, val, pos) in enumerate(zip(preds, probs_or_presence, pred_pos)):
        hemi = "LEFT" if pos[0] < 0 else "RIGHT"
        print(f"  slot{slot}: region={region}  {label}={val:.4f}  pos_mm={pos}  ({hemi} hemisphere)")

    print(f"\nReference eLORETA estimate (source idx {gt['source_index_1based']}, in the "
         f"subject's own coordinate frame): pos_mm={gt['coord_mm']}  (LEFT hemisphere, "
         f"dorsal/central -- M1 hand-area, consistent with a right-hand button press)")
    print("NOTE: that coordinate frame is not verified to align with Ernie's -- treat any "
         "Euclidean distance below as a rough proxy, not a validated localization error:")
    for slot, pos in enumerate(pred_pos):
        dist = np.linalg.norm(pos - gt["coord_mm"])
        print(f"  slot{slot} distance to reference coordinate: {dist:.1f}mm (rough proxy only)")


def run_infer_motor(checkpoint, clusters="10mm", noise_level=0.50, num_sources=2, pe="none",
                    dropout_num=0, subject="sub_001", proxy_montage_subject=PROXY_MONTAGE_SUBJECT):
    import torch
    from helpers import load_data, load_locs, load_dist_table
    from ablation import LitMultiEmbedNSource

    ernie_signal, ernie_noise_std, gt, electrode_locs = _prepare_real_input(
        clusters, subject, proxy_montage_subject)

    leadfield, _ = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    ref_norm = _training_scale_reference(clusters, noise_level, num_sources)
    raw_norm = np.linalg.norm(ernie_signal)
    scale = ref_norm / raw_norm
    scaled_signal = ernie_signal * scale
    snr = np.linalg.norm(ernie_signal) / np.linalg.norm(ernie_noise_std)

    print(f"[{subject}] raw ||signal||_2={raw_norm:.3e}V, rescaled by {scale:.3e} to match "
         f"training distribution (new norm={np.linalg.norm(scaled_signal):.2f})")
    print(f"[{subject}] real single-trial noise ||std||_2={np.linalg.norm(ernie_noise_std):.3e}V "
         f"(trial-to-trial std in the response window -- see module docstring's REAL NOISE note) "
         f"-- signal/noise norm ratio = {snr:.2f}")

    model = LitMultiEmbedNSource.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        num_sources=num_sources, cluster_locs=cluster_locs, dist_table=dist_table, pe=pe,
        dropout_num=dropout_num,
    )
    model.eval()

    signal_t = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model.model(signal_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    preds = probs.argmax(axis=1)
    pred_pos = cluster_locs[preds]
    slot_probs = probs[np.arange(num_sources), preds]

    _report_prediction(subject, preds, slot_probs, pred_pos, gt, label="p")

    out_path = "real_data_experiment_motor_output.npz"
    np.savez(out_path, preds=preds, probs=probs, pred_pos_mm=pred_pos, scale_factor=scale,
            snr=snr, gt_coord_mm=gt["coord_mm"])
    print(f"\nWrote {out_path}")
    return preds, pred_pos, gt


def run_infer_motor_objectness(checkpoint, clusters="10mm", noise_level=0.50, min_sources=1,
                               max_sources=5, presence_threshold=0.5, pe="none", dropout_num=0,
                               subject="sub_001", proxy_montage_subject=PROXY_MONTAGE_SUBJECT):
    """Variable/unknown-count counterpart of run_infer_motor -- same cross-check run on the
    face-processing data: does a fixed 2-source assumption explain the implausible slot1, or does
    the objectness model (free to say '1') do the same thing? See real_data_experiment.py's
    MODEL CHOICE docstring section for the full rationale."""
    import torch
    from helpers import load_data, load_locs, load_dist_table
    from ablation_unknown_count import LitObjectnessNSource

    ernie_signal, ernie_noise_std, gt, electrode_locs = _prepare_real_input(
        clusters, subject, proxy_montage_subject)

    leadfield, _ = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    ref_norm = _training_scale_reference_variable(clusters, noise_level, min_sources, max_sources)
    raw_norm = np.linalg.norm(ernie_signal)
    scale = ref_norm / raw_norm
    scaled_signal = ernie_signal * scale
    snr = np.linalg.norm(ernie_signal) / np.linalg.norm(ernie_noise_std)

    print(f"[{subject}] raw ||signal||_2={raw_norm:.3e}V, rescaled by {scale:.3e} to match "
         f"training distribution (new norm={np.linalg.norm(scaled_signal):.2f})")
    print(f"[{subject}] real single-trial noise ||std||_2={np.linalg.norm(ernie_noise_std):.3e}V "
         f"-- signal/noise norm ratio = {snr:.2f}")

    model = LitObjectnessNSource.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        max_sources=max_sources, cluster_locs=cluster_locs, dist_table=dist_table, pe=pe,
        dropout_num=dropout_num, presence_threshold=presence_threshold,
    )
    model.eval()

    signal_t = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        loc_logits, presence_logits = model.model(signal_t)
        loc_probs = torch.softmax(loc_logits, dim=-1).squeeze(0).numpy()
        presence_probs = torch.sigmoid(presence_logits).squeeze(0).numpy()
    preds = loc_probs.argmax(axis=1)
    pred_pos = cluster_locs[preds]
    active = presence_probs > presence_threshold

    print(f"\n[{subject}] {int(active.sum())}/{max_sources} slot(s) cleared "
         f"presence_threshold={presence_threshold}:")
    for slot in range(max_sources):
        flag = "ACTIVE" if active[slot] else "  --  "
        hemi = "LEFT" if pred_pos[slot, 0] < 0 else "RIGHT"
        print(f"  slot{slot} [{flag}]: presence_p={presence_probs[slot]:.4f}  region={preds[slot]}  "
             f"loc_p={loc_probs[slot, preds[slot]]:.4f}  pos_mm={pred_pos[slot]}  ({hemi})")

    active_preds = preds[active]
    active_pos = pred_pos[active]
    if len(active_preds) > 0:
        print(f"\n-- Active-slot-only comparison to ground truth --")
        _report_prediction(subject, active_preds, presence_probs[active], active_pos, gt,
                           label="presence_p")
    else:
        print(f"\n[{subject}] no slots cleared presence_threshold={presence_threshold} -- "
             f"model reports zero sources present.")

    out_path = "real_data_experiment_motor_objectness_output.npz"
    np.savez(out_path, preds=preds, loc_probs=loc_probs, presence_probs=presence_probs,
            active=active, pred_pos_mm=pred_pos, scale_factor=scale, snr=snr,
            gt_coord_mm=gt["coord_mm"])
    print(f"\nWrote {out_path}")
    return preds, pred_pos, active, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a trained Ernie checkpoint to real motor "
                                                 "(button-press) EEG, no retraining")
    parser.add_argument("--model", type=str, default="multiembed", choices=["multiembed", "objectness"],
                        help="multiembed: fixed --num_sources slots (ablation.LitMultiEmbedNSource). "
                             "objectness: variable/unknown count with a per-slot presence head "
                             "(ablation_unknown_count.LitObjectnessNSource).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint matching --model")
    parser.add_argument("--clusters", type=str, default="10mm")
    parser.add_argument("--noise_level", type=float, default=0.50)
    parser.add_argument("--num_sources", type=int, default=2, help="--model multiembed only")
    parser.add_argument("--min_sources", type=int, default=1, help="--model objectness only")
    parser.add_argument("--max_sources", type=int, default=5, help="--model objectness only")
    parser.add_argument("--presence_threshold", type=float, default=0.5, help="--model objectness only")
    parser.add_argument("--pe", type=str, default="none")
    parser.add_argument("--dropout_num", type=int, default=0)
    parser.add_argument("--subject", type=str, default="sub_001")
    parser.add_argument("--proxy_montage_subject", type=str, default=PROXY_MONTAGE_SUBJECT,
                        help="ds003645 subject whose electrodes.tsv is borrowed as the electrode "
                             "positions (subject 001's own digitization is not available; see "
                             "module docstring)")
    args = parser.parse_args()

    if args.model == "objectness":
        run_infer_motor_objectness(args.checkpoint, args.clusters, args.noise_level,
                                   args.min_sources, args.max_sources, args.presence_threshold,
                                   args.pe, args.dropout_num, args.subject, args.proxy_montage_subject)
    else:
        run_infer_motor(args.checkpoint, args.clusters, args.noise_level, args.num_sources, args.pe,
                        args.dropout_num, args.subject, args.proxy_montage_subject)
