"""
ablation_unknown_count.py -- what if the number of active sources isn't known in advance?

Both ablation.py models so far were told n at inference time (a fixed top-n / fixed number of
query slots). This script drops that assumption: every sample independently has a random
number of simultaneous sources drawn from [min_sources, max_sources] (default 1-5, equal
strength -- see VariableNSourceDataset), and *neither* model is given that count at test time.
Each model has to recover both "how many" and "where" from the signal alone.

Two count-free decoders are compared, corresponding to the two directions from the design
discussion this script follows up on:

  1. `LitThresholdBaseline` -- the single dense "heatmap" head from ablation.py
     (TopKHeatmapNetwork, reused as-is), but decoded by thresholding sigmoid(logits) instead
     of a fixed top-k. This is how the single-heatmap parameterization has actually been used
     count-free in the literature: ConvDip (Hecker et al., 2021, Frontiers in Neuroscience)
     explicitly frames this as its advantage over earlier fixed-slot networks ("the output
     vector...only contained the position of one, two, or three sources, meaning the number of
     expected sources must be known in advance"); classical (pre-DL) EEG/MEG source imaging
     (LORETA/sLORETA, beamformers) has long thresholded a continuous activation map at some
     percentage of its peak amplitude rather than assuming a known source count.
  2. `LitObjectnessNSource` -- our per-source query-embedding method, extended with an
     explicit presence/"objectness" output per slot, trained end-to-end against a padded
     "no object" class -- i.e. DETR's (Carion et al., 2020, arXiv:2005.12872) recipe for
     variable-cardinality set prediction, applied to our transformer backbone. A fixed
     `max_sources` query slots are always run; the matching loss (`_objectness_matching_loss`)
     optimally assigns each slot to either a real ground-truth source (location CE + presence
     loss) or to an implicit "empty" label (presence loss only), via bipartite (Hungarian)
     matching over the padded label set -- generalizing ablation.py's
     `_permutation_invariant_ce` from "always exactly n real slots" to "up to max_sources
     slots, some of which may be empty." At inference, slots whose presence probability clears
     `--presence_threshold` are kept; the survivors' argmax locations are the predicted set.

Because neither model is given the count, "accuracy against a known n" no longer makes sense.
Evaluation instead uses precision/recall/F1 over the predicted location *set* against the true
set (a predicted location that doesn't exist is now a false positive, exactly the shift
object-detection metrics made going from classification to detection), plus count MAE/accuracy
and the same optimally-matched Euclidean/geodesic localization error used elsewhere in this
project.

Usage:
    python ablation_unknown_count.py 10mm 0.50 --min_sources 1 --max_sources 5 --method both

References
----------
- Hecker, Rupprecht, Elst, Kornmeier (2021). ConvDip: A Convolutional Neural Network for
  Better EEG Source Imaging. Frontiers in Neuroscience, 15:569918.
- Carion et al. (2020). End-to-End Object Detection with Transformers (DETR).
  arXiv:2005.12872. Source of the padded "no object" class / bipartite matching loss that
  `_objectness_matching_loss` below generalizes our permutation-invariant loss with.
- Zhang, Hare, Prügel-Bennett (2019). Deep Set Prediction Networks. NeurIPS. arXiv:1906.06565.
- On the Estimation of the Number of Dipole Sources in EEG Source Localization (PMC1945217) and
  the F-ratio-based method for MEG (arXiv:2306.05892) -- classical (AIC/BIC/RAP-MUSIC-family)
  antecedents of "figure out how many sources from the data itself."
"""

import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from helpers import load_data, getNoiseVectors, load_dist_table, load_locs, load_electrode_locs
from clusterNetworks import _drop_electrode_tokens
from positional_encoding_3d import PositionalEncoding3D
from ablation import TopKHeatmapNetwork, _BaseAblationModule, nms_select


PAD_LOCATION = -1


# ============================================================================
# Data: random source count per sample, equal-strength mixing
# ============================================================================

class VariableNSourceDataset(Dataset):
    """
    Equal-strength mixture dataset with a *random* source count n drawn fresh per sample from
    [min_sources, max_sources], instead of a fixed n (generalizes NSourceDataset/
    MultiClusterDataset, which always used a fixed, known n).

    Equal-strength convention: sources are simply summed with equal weight (no alpha ratio) --
    this is exactly the alpha=0.5 special case of MultiClusterDataset's mixing rule ("reduces
    to lf0 + lf1, matching the equal-strength baseline" -- myDatasets.py).

    Returns (signal, clusters, n) where clusters is a LongTensor of length max_sources, with
    entries [n:] set to PAD_LOCATION (-1), and n is a scalar long tensor giving the true count.
    """

    def __init__(self, num_dipoles, leadfield, num_examples, min_sources=1, max_sources=5,
                 dist_table=None, min_dist=20.0, noise_vectors=None, noise_level=0.0,
                 noise_mode="sensor"):
        super().__init__()
        if min_sources < 1 or max_sources < min_sources:
            raise ValueError("Require 1 <= min_sources <= max_sources")
        self.num_dipoles = num_dipoles
        self.leadfield = leadfield
        self.num_examples = num_examples
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.dist_table = dist_table
        self.min_dist = min_dist if dist_table is not None else 0.0
        self.noise_mode = noise_mode
        self.length = num_examples * num_dipoles

        if noise_vectors is None:
            self.noise_vectors = None
            noise_max = np.amax(np.abs(leadfield), axis=0)
            self.std_devs = noise_max / 2 * noise_level
        else:
            self.noise_vectors = torch.tensor(noise_vectors, dtype=torch.float32)

    def _sample_clusters(self, n):
        pairs = list(itertools.combinations(range(n), 2))
        candidate = None
        for _ in range(1000):
            candidate = np.random.choice(self.num_dipoles, n, replace=False)
            if self.dist_table is None or not pairs:
                return candidate
            if all(self.dist_table[candidate[i], candidate[j]] >= self.min_dist for i, j in pairs):
                return candidate
        return candidate  # best-effort fallback if no valid draw found in 1000 tries

    def __getitem__(self, idx):
        example_idx = idx // self.num_dipoles
        n = int(np.random.randint(self.min_sources, self.max_sources + 1))
        clusters = self._sample_clusters(n)

        lf = torch.zeros(self.leadfield.shape[0], dtype=torch.float32)
        for c in clusters:
            lf = lf + torch.tensor(self.leadfield[:, int(c)], dtype=torch.float32)

        c0 = int(clusters[0])
        if self.noise_vectors is None:
            noise = torch.normal(0.0, float(self.std_devs[c0]), size=lf.shape)
        else:
            ni = example_idx % self.noise_vectors.shape[0]
            noise = self.noise_vectors[ni, :, c0]

        padded = torch.full((self.max_sources,), PAD_LOCATION, dtype=torch.long)
        padded[:n] = torch.tensor(clusters, dtype=torch.long)

        return lf + noise, padded, torch.tensor(n, dtype=torch.long)

    def __len__(self):
        return self.length


# ============================================================================
# Model 2: per-slot location + presence ("objectness") transformer
# ============================================================================

class ObjectnessTransformerEncoder(nn.Module):
    """
    Same electrode tokenization and per-source learned-query design as
    clusterNetworks.TransformerEncoder, but with `max_sources` always-run query slots (a fixed
    upper bound, not the true count) and a second output head per slot: a presence/"objectness"
    logit alongside the usual location logits. A slot with low presence probability means
    "no source here" -- the DETR "no object" class applied to our query-embedding design.
    """

    def __init__(self, in_channels, out_channels, electrode_locs, max_sources=5, d_model=512,
                 nhead=8, num_encoder_layers=3, with_pos_enc=False, dropout=0.1, dropout_num=0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.max_sources = max_sources
        self.dropout_num = int(dropout_num)
        self.num_keep = in_channels - self.dropout_num
        if self.num_keep <= 0:
            raise ValueError(f"dropout_num={dropout_num} must be less than in_channels={in_channels}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.with_pos_enc = with_pos_enc

        if with_pos_enc:
            self.input_projection = nn.Linear(1, d_model)
            self.pe = PositionalEncoding3D(d_model)
        else:
            self.input_projection = nn.Linear(4, d_model)

        self.register_buffer("electrode_locs", torch.tensor(electrode_locs, dtype=torch.float32))
        self.learned_tokens = nn.Parameter(torch.rand(max_sources, d_model))
        self.location_head = nn.Linear(d_model, out_channels)
        self.presence_head = nn.Linear(d_model, 1)

    def forward(self, x):
        if self.with_pos_enc:
            x_in = x.unsqueeze(2)
            x_in = self.input_projection(x_in)
            encodings = self.pe(self.electrode_locs).unsqueeze(0).expand(x_in.size(0), -1, -1)
            x_tokens = x_in + encodings
        else:
            x_tokens = self.input_projection(
                torch.cat((x.unsqueeze(2), self.electrode_locs.expand(x.shape[0], -1, -1)), dim=2)
            )

        if self.dropout_num > 0:
            x_tokens = _drop_electrode_tokens(x_tokens, self.num_keep)

        tokens = self.learned_tokens.unsqueeze(0).expand(x_tokens.size(0), -1, -1)
        x_out = self.transformer_encoder(torch.cat((tokens, x_tokens), dim=1))

        slot_hidden = x_out[:, :self.max_sources, :]
        loc_logits = self.location_head(slot_hidden)              # (B, max_sources, C)
        presence_logits = self.presence_head(slot_hidden).squeeze(-1)  # (B, max_sources)
        return loc_logits, presence_logits


def _objectness_matching_cost(loc_logits, presence_logits, labels, n_true, loc_weight, presence_weight):
    """
    Builds the (B, max_sources, max_sources) cost of assigning predicted slot i to padded
    label column j. Columns j < n_true[b] are real sources (location CE + presence-vs-1 cost);
    columns j >= n_true[b] are the implicit "empty" label (presence-vs-0 cost only, no
    location cost -- there's nothing for a location to be correct *about* there).
    """
    B, N_max, C = loc_logits.shape
    device = loc_logits.device

    col_idx = torch.arange(N_max, device=device).unsqueeze(0).expand(B, -1)
    real_mask = (col_idx < n_true.unsqueeze(1)).float()  # (B, N_max) over label columns j
    safe_labels = labels.clamp(min=0)                    # PAD_LOCATION -> 0, masked out below

    logp = torch.log_softmax(loc_logits, dim=-1)  # (B, N_max_i, C)
    label_idx = safe_labels.unsqueeze(1).expand(B, N_max, N_max)  # (B, i, j) -> gather over C
    loc_cost = -torch.gather(logp, dim=2, index=label_idx)        # (B, i, j)
    loc_cost = loc_cost * real_mask.unsqueeze(1)                  # zero cost vs. empty columns

    target = real_mask.unsqueeze(1).expand(B, N_max, N_max)       # (B, i, j)
    pred = presence_logits.unsqueeze(2).expand(B, N_max, N_max)   # (B, i, j)
    presence_cost = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    return loc_weight * loc_cost + presence_weight * presence_cost


def _objectness_matching_loss(loc_logits, presence_logits, labels, n_true,
                              loc_weight=1.0, presence_weight=1.0):
    """
    Optimal (Hungarian) bipartite matching of predicted slots to the padded label set using
    the cost above, then the mean per-slot cost at that matching -- the generalization of
    ablation.py's `_permutation_invariant_ce` to a variable, unknown number of real sources.

    Returns (loss: scalar, assignment: (B, max_sources) long tensor of matched label columns).
    """
    cost = _objectness_matching_cost(loc_logits, presence_logits, labels, n_true, loc_weight, presence_weight)
    B, N_max, _ = cost.shape
    cost_np = cost.detach().cpu().numpy()
    assignment = torch.empty(B, N_max, dtype=torch.long, device=loc_logits.device)
    total = torch.zeros((), device=loc_logits.device)
    for b in range(B):
        row, col = linear_sum_assignment(cost_np[b])
        assignment[b, row] = torch.tensor(col, device=loc_logits.device)
        total = total + cost[b, row, col].sum()
    return total / (B * N_max), assignment


def _decode_objectness_sets(loc_logits, presence_logits, threshold=0.5):
    """Slots clearing the presence threshold survive; their argmax locations are the predicted
    set (de-duplicated, since two slots could in principle argmax to the same location)."""
    probs = torch.sigmoid(presence_logits)
    preds = torch.argmax(loc_logits, dim=-1)
    pred_sets = []
    for b in range(probs.shape[0]):
        active = (probs[b] > threshold).nonzero(as_tuple=True)[0]
        locs = preds[b, active]
        pred_sets.append(np.unique(locs.cpu().numpy()))
    return pred_sets


def _true_sets_from_padded(labels, n_true):
    true_sets = []
    for b in range(labels.shape[0]):
        n = int(n_true[b].item())
        true_sets.append(labels[b, :n].cpu().numpy())
    return true_sets


# ============================================================================
# Shared evaluation: variable-size predicted set vs. variable-size true set
# ============================================================================

def _nn_dists_rows(cost_matrix):
    """For each row, the min over columns (nearest-neighbor distance from that row's point to
    the closest column-side point), excluding rows that are entirely non-finite (e.g.
    geodesically unreachable from every point on the other side)."""
    finite = np.isfinite(cost_matrix)
    row_has_finite = finite.any(axis=1)
    safe = np.where(finite, cost_matrix, np.inf)
    return safe.min(axis=1)[row_has_finite]


def _nn_dists_cols(cost_matrix):
    """Column-wise counterpart of _nn_dists_rows."""
    finite = np.isfinite(cost_matrix)
    col_has_finite = finite.any(axis=0)
    safe = np.where(finite, cost_matrix, np.inf)
    return safe.min(axis=0)[col_has_finite]


def _variable_match_metrics(pred_sets, true_sets, cluster_locs=None, dist_table=None):
    """
    pred_sets, true_sets: lists (length B) of 1D int arrays of location indices -- possibly
    different lengths per sample (including an empty predicted set). Since neither model knows
    n, plain "accuracy" no longer applies; this reports set precision/recall/F1 (a predicted
    location outside the true set is now a false positive, the same shift object-detection
    metrics made over plain classification), count error, and two different flavors of
    localization error -- see the long comment below on `euclid_dist`/`geodist` vs.
    `chamfer_p2t_*`/`chamfer_t2p_*` before trusting either one on its own.

    Also reports `hit_rate` and `exact_match`, matching the names used in ablation.py's
    `_bipartite_match_metrics`, for direct cross-script comparability:
      - `hit_rate` = |pred set (cap) true set| / |true set| = recall as defined above. In
        ablation.py, |pred| was always forced equal to |true| (fixed, known n decoded via
        top-n), which made precision, recall, and hit_rate numerically identical there. Here
        |pred| and |true| can differ, so hit_rate and recall diverge from precision; hit_rate
        is kept as an explicit alias of recall rather than a new quantity, so the name means
        the same thing as it did in the first ablation.
      - `exact_match` = whether the predicted set equals the true set exactly (same elements,
        which also implies the same count) -- the direct generalization of the original
        `val_acc`/`test_acc` "all sources correctly matched" metric from main.py.

    LOCALIZATION ERROR: bipartite (`euclid_dist`/`geodist`) vs. Chamfer (`chamfer_*`)
    -----------------------------------------------------------------------------
    `euclid_dist`/`geodist` use optimal (Hungarian) bipartite matching, which on a rectangular
    cost matrix only produces min(|pred|, |true|) matched pairs -- every *unmatched* prediction
    is silently dropped from the average, not penalized. A method that over-predicts gets to
    cherry-pick its best min(|pred|,|true|) predictions for this metric and pay nothing for the
    rest; concretely, predicting the exactly-right location plus three wildly wrong extra ones
    scores an identical 0.0 to predicting only the right one. That's a real blind spot, not a
    rounding error, and it specifically flatters methods (like the threshold-decoded heatmap
    baseline) that tend to over-predict.

    `chamfer_p2t_euclid`/`chamfer_p2t_geo` (predicted -> true) fix exactly this: for *every*
    predicted point, not just a matched subset, find its distance to the nearest true source,
    then average over all of them. A spurious extra prediction now always adds its own
    (typically large) distance into the average, so over-prediction is no longer free. This is
    standard Chamfer-distance practice from the point-set comparison literature (e.g. Fan,
    Su & Guibas, 2017, "A Point Set Generation Network for 3D Object Reconstruction from a
    Single Image") applied to source locations instead of 3D points, and it's the metric to
    use when comparing methods that predict different numbers of sources.

    `chamfer_t2p_euclid`/`chamfer_t2p_geo` (true -> predicted) is the other Chamfer direction --
    for every true source, distance to the nearest prediction. Reported for completeness, but
    it does NOT fix the over-prediction bias: more candidate predictions can only make the
    nearest one closer, so this direction still rewards over-predicting. Don't cite it as the
    "fixed" metric; `chamfer_p2t_*` is the one that is.
    """
    precisions, recalls, f1s, count_err, count_acc, exact_matches = [], [], [], [], [], []
    euclid_vals, geo_vals = [], []
    chamfer_p2t_euclid, chamfer_t2p_euclid, chamfer_p2t_geo, chamfer_t2p_geo = [], [], [], []

    for pred, true in zip(pred_sets, true_sets):
        pred_set, true_set = set(pred.tolist()), set(true.tolist())
        tp = len(pred_set & true_set)
        precision = (tp / len(pred_set)) if len(pred_set) > 0 else (1.0 if len(true_set) == 0 else 0.0)
        recall = (tp / len(true_set)) if len(true_set) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        count_err.append(abs(len(pred) - len(true)))
        count_acc.append(float(len(pred) == len(true)))
        exact_matches.append(float(pred_set == true_set))

        if len(pred) > 0 and len(true) > 0:
            if cluster_locs is not None:
                cost = torch.cdist(cluster_locs[pred], cluster_locs[true]).cpu().numpy()
                row, col = linear_sum_assignment(cost)
                euclid_vals.append(cost[row, col].mean())

                p2t = _nn_dists_rows(cost)  # per predicted point -- euclidean is always finite
                chamfer_p2t_euclid.append(float(p2t.mean()))
                t2p = _nn_dists_cols(cost)  # per true point
                chamfer_t2p_euclid.append(float(t2p.mean()))

            if dist_table is not None:
                cost_np = dist_table[np.ix_(pred, true)]
                if torch.is_tensor(cost_np):
                    cost_np = cost_np.cpu().numpy()
                finite_mask = np.isfinite(cost_np)
                cost_for_match = np.where(finite_mask, cost_np, 1e6)
                row, col = linear_sum_assignment(cost_for_match)
                matched = cost_np[row, col]
                matched = matched[np.isfinite(matched)]
                if matched.size:
                    geo_vals.append(matched.mean())

                p2t_geo = _nn_dists_rows(cost_np)
                if p2t_geo.size:
                    chamfer_p2t_geo.append(float(p2t_geo.mean()))
                t2p_geo = _nn_dists_cols(cost_np)
                if t2p_geo.size:
                    chamfer_t2p_geo.append(float(t2p_geo.mean()))

    out = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "hit_rate": float(np.mean(recalls)),  # alias of recall here -- see docstring
        "f1": float(np.mean(f1s)),
        "exact_match": float(np.mean(exact_matches)),
        "count_mae": float(np.mean(count_err)),
        "count_acc": float(np.mean(count_acc)),
    }
    if euclid_vals:
        out["euclid_dist"] = float(np.mean(euclid_vals))
    if geo_vals:
        out["geodist"] = float(np.mean(geo_vals))
    if chamfer_p2t_euclid:
        out["chamfer_p2t_euclid"] = float(np.mean(chamfer_p2t_euclid))
    if chamfer_t2p_euclid:
        out["chamfer_t2p_euclid"] = float(np.mean(chamfer_t2p_euclid))
    if chamfer_p2t_geo:
        out["chamfer_p2t_geo"] = float(np.mean(chamfer_p2t_geo))
    if chamfer_t2p_geo:
        out["chamfer_t2p_geo"] = float(np.mean(chamfer_t2p_geo))
    return out


class _BaseUnknownCountModule(_BaseAblationModule):
    def _log_match_metrics(self, split, pred_sets, true_sets):
        metrics = _variable_match_metrics(pred_sets, true_sets, self.cluster_locs, self.dist_table)
        for name, value in metrics.items():
            prog = name in ("f1", "recall", "count_mae")
            self.log(f"{split}_{name}", value, batch_size=self.b, prog_bar=prog,
                      on_step=False, on_epoch=True, sync_dist=True)


# ============================================================================
# LightningModule 1: heatmap + count-free threshold decode
# ============================================================================

class LitThresholdBaseline(_BaseUnknownCountModule):
    def __init__(self, in_channels, out_channels, electrode_locs, min_sources=1, max_sources=5,
                 batch_size=512, lr=1e-4, max_epochs=200, cluster_locs=None, dist_table=None,
                 scheduler="onecycle", pe="none", prob_threshold=0.5, pos_weight_scale=1.0,
                 dropout_num=0, nms_radius=20.0):
        super().__init__(batch_size, lr, max_epochs, cluster_locs, dist_table, electrode_locs, scheduler)
        self.out_channels = out_channels
        self.prob_threshold = prob_threshold
        # Suppression radius (mm, geodesic) for the eval-time NMS decode below; 0 recovers
        # plain thresholding. See ablation.nms_select's docstring for why this only affects
        # decoding, not loss.
        self.nms_radius = nms_radius

        with_pos_enc = pe in ("enc", "both")
        self.model = TopKHeatmapNetwork(in_channels, out_channels, electrode_locs,
                                        with_pos_enc=with_pos_enc, dropout_num=dropout_num)

        # We no longer know the exact positive count per sample ahead of time (that's the
        # point), so pos_weight is balanced against the *expected* count, not an exact one.
        avg_n = (min_sources + max_sources) / 2.0
        pos_weight = torch.tensor((out_channels - avg_n) / max(avg_n, 1e-6) * pos_weight_scale)
        self.register_buffer("pos_weight", pos_weight)
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def _make_targets(self, labels, n_true):
        # Route padded (invalid) columns to an extra dummy class so they can never collide
        # with a real location index (labels are clamped to 0 there, which would otherwise
        # falsely zero out a genuine positive at location 0 when scattered directly).
        B, N_max = labels.shape
        col_idx = torch.arange(N_max, device=labels.device).unsqueeze(0).expand(B, -1)
        valid = col_idx < n_true.unsqueeze(1)
        dummy = self.out_channels
        index = torch.where(valid, labels.clamp(min=0), torch.full_like(labels, dummy))
        target_full = torch.zeros(B, self.out_channels + 1, device=labels.device)
        target_full.scatter_(1, index, 1.0)
        return target_full[:, :self.out_channels]

    def _step(self, data, split):
        leadfield, labels, n_true = data[0], data[1], data[2]
        logits = self.model(leadfield)
        target = self._make_targets(labels, n_true)
        loss = self.loss_func(logits, target)
        self.log(f"{split}_loss", loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if split != "train":
            probs = torch.sigmoid(logits)
            pred_sets = nms_select(probs, self.dist_table, self.nms_radius, threshold=self.prob_threshold)
            true_sets = _true_sets_from_padded(labels, n_true)
            self._log_match_metrics(split, pred_sets, true_sets)

        return loss

    def training_step(self, data, batch_idx):
        return self._step(data, "train")

    def validation_step(self, data, batch_idx):
        return self._step(data, "val")

    def test_step(self, data, batch_idx):
        return self._step(data, "test")


# ============================================================================
# LightningModule 2: per-slot location + objectness, DETR-style matching
# ============================================================================

class LitObjectnessNSource(_BaseUnknownCountModule):
    def __init__(self, in_channels, out_channels, electrode_locs, max_sources=5, batch_size=512,
                 lr=1e-4, max_epochs=200, cluster_locs=None, dist_table=None, scheduler="onecycle",
                 pe="none", presence_threshold=0.5, loc_weight=1.0, presence_weight=1.0, dropout_num=0):
        super().__init__(batch_size, lr, max_epochs, cluster_locs, dist_table, electrode_locs, scheduler)
        self.max_sources = max_sources
        self.presence_threshold = presence_threshold
        self.loc_weight = loc_weight
        self.presence_weight = presence_weight

        with_pos_enc = pe in ("enc", "both")
        self.model = ObjectnessTransformerEncoder(in_channels, out_channels, electrode_locs,
                                                   max_sources=max_sources, with_pos_enc=with_pos_enc,
                                                   dropout_num=dropout_num)

    def _step(self, data, split):
        leadfield, labels, n_true = data[0], data[1], data[2]
        loc_logits, presence_logits = self.model(leadfield)
        loss, _assignment = _objectness_matching_loss(loc_logits, presence_logits, labels, n_true,
                                                       self.loc_weight, self.presence_weight)
        self.log(f"{split}_loss", loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if split != "train":
            pred_sets = _decode_objectness_sets(loc_logits, presence_logits, self.presence_threshold)
            true_sets = _true_sets_from_padded(labels, n_true)
            self._log_match_metrics(split, pred_sets, true_sets)

        return loss

    def training_step(self, data, batch_idx):
        return self._step(data, "train")

    def validation_step(self, data, batch_idx):
        return self._step(data, "val")

    def test_step(self, data, batch_idx):
        return self._step(data, "test")


# ============================================================================
# Driver
# ============================================================================

def _build_loaders(clusters, noise_level, min_sources, max_sources, min_dist, batch_size, workers,
                   train_size=100, val_size=50, test_size=100):
    leadfield, noise_dir = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    num_dipoles = leadfield.shape[1]

    val_noise, test_noise = getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True)

    train_dataset = VariableNSourceDataset(num_dipoles, leadfield, train_size, min_sources=min_sources,
                                           max_sources=max_sources, dist_table=dist_table, min_dist=min_dist,
                                           noise_level=noise_level)
    val_dataset = VariableNSourceDataset(num_dipoles, leadfield, val_size, min_sources=min_sources,
                                        max_sources=max_sources, dist_table=dist_table, min_dist=min_dist,
                                        noise_vectors=val_noise)
    test_dataset = VariableNSourceDataset(num_dipoles, leadfield, test_size, min_sources=min_sources,
                                         max_sources=max_sources, dist_table=dist_table, min_dist=min_dist,
                                         noise_vectors=test_noise)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    return train_loader, val_loader, test_loader, leadfield, cluster_locs, dist_table, electrode_locs


def _make_logger(use_tensorboard, log_dir, log_name, wandb_project, wandb_entity, group, tags):
    if use_tensorboard:
        return pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name)
    return pl_loggers.WandbLogger(project=wandb_project, entity=wandb_entity, name=log_name,
                                  save_dir=log_dir, group=group, tags=tags)


def _train_and_test(model, train_loader, val_loader, test_loader, max_epochs, progress_bar, log_name, log_dir,
                    use_tensorboard=False, wandb_project=None, wandb_entity=None, group=None, tags=None):
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_f1", save_top_k=1, mode="max")
    logger = _make_logger(use_tensorboard, log_dir, log_name, wandb_project, wandb_entity, group, tags)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=[checkpoint],
                         logger=logger, enable_progress_bar=progress_bar)
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    if not use_tensorboard:
        import wandb  # only needed (and only imported) when W&B logging is in use
        wandb.finish()  # close this run so the next model in the same process starts a fresh one
    return results[0] if results else {}


def run_ablation(clusters, noise_level, min_sources=1, max_sources=5, method="both", batch_size=512,
                 max_epochs=200, learning_rate=1e-4, scheduler="onecycle", pe="none", workers=4,
                 progress_bar=True, min_dist=20.0, prob_threshold=0.5, presence_threshold=0.5,
                 pos_weight_scale=1.0, loc_weight=1.0, presence_weight=1.0, dropout_num=0,
                 nms_radius=20.0, use_tensorboard=False, wandb_project=None, wandb_entity=None):

    (train_loader, val_loader, test_loader, leadfield,
     cluster_locs, dist_table, electrode_locs) = _build_loaders(
        clusters, noise_level, min_sources, max_sources, min_dist, batch_size, workers)

    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]
    s = "{s:.2f}".format(s=noise_level).split(".")[1]
    log_dir = f"my_logs/ablation_unknown_count_cluster{clusters}_{min_sources}to{max_sources}src/noise_{s}"
    group = f"{clusters}_{min_sources}to{max_sources}src_noise{s}"
    wandb_project = wandb_project or "eeg-ablation-unknown-count"

    results = {}

    if method in ("topk", "both"):
        topk_model = LitThresholdBaseline(in_channels, out_channels, electrode_locs,
                                          min_sources=min_sources, max_sources=max_sources,
                                          batch_size=batch_size, lr=learning_rate, max_epochs=max_epochs,
                                          cluster_locs=cluster_locs, dist_table=dist_table, scheduler=scheduler,
                                          pe=pe, prob_threshold=prob_threshold, pos_weight_scale=pos_weight_scale,
                                          dropout_num=dropout_num, nms_radius=nms_radius)
        log_name = f"ThresholdHeatmap_{scheduler}_{max_epochs}_{learning_rate:.0e}"
        results["threshold_heatmap_baseline"] = _train_and_test(
            topk_model, train_loader, val_loader, test_loader, max_epochs, progress_bar, log_name, log_dir,
            use_tensorboard=use_tensorboard, wandb_project=wandb_project, wandb_entity=wandb_entity,
            group=group, tags=["threshold_heatmap_baseline"])

    if method in ("objectness", "both"):
        obj_model = LitObjectnessNSource(in_channels, out_channels, electrode_locs, max_sources=max_sources,
                                         batch_size=batch_size, lr=learning_rate, max_epochs=max_epochs,
                                         cluster_locs=cluster_locs, dist_table=dist_table, scheduler=scheduler,
                                         pe=pe, presence_threshold=presence_threshold, loc_weight=loc_weight,
                                         presence_weight=presence_weight, dropout_num=dropout_num)
        log_name = f"Objectness_{scheduler}_{max_epochs}_{learning_rate:.0e}"
        results["objectness_multi_embedding"] = _train_and_test(
            obj_model, train_loader, val_loader, test_loader, max_epochs, progress_bar, log_name, log_dir,
            use_tensorboard=use_tensorboard, wandb_project=wandb_project, wandb_entity=wandb_entity,
            group=group, tags=["objectness_multi_embedding"])

    if results:
        _print_comparison(results, min_sources, max_sources)

    return results


def evaluate_checkpoint(clusters, noise_level, checkpoint, min_sources=1, max_sources=5,
                        nms_radius=20.0, pe="none", dropout_num=0, batch_size=512, workers=4,
                        progress_bar=True, test_size=100, val_size=50):
    """
    Re-evaluate an already-trained LitThresholdBaseline checkpoint on a fresh test-set draw
    with a chosen nms_radius -- no retraining. See ablation.evaluate_checkpoint (same idea,
    for the fixed-count baseline) and nms_select's docstring for why this is safe to do without
    touching the trained weights.

    Note: this test set is a fresh random draw (only the additive noise is cached/reproducible
    -- the source counts/locations themselves are re-sampled), so the numbers are a new,
    independent sample from the same distribution, not a literal replay of the original
    trainer.test() call.
    """
    leadfield, noise_dir = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    num_dipoles = leadfield.shape[1]
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    _, test_noise = getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True)
    test_dataset = VariableNSourceDataset(num_dipoles, leadfield, test_size, min_sources=min_sources,
                                         max_sources=max_sources, dist_table=dist_table,
                                         noise_vectors=test_noise)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    model = LitThresholdBaseline.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        min_sources=min_sources, max_sources=max_sources, cluster_locs=cluster_locs,
        dist_table=dist_table, pe=pe, dropout_num=dropout_num, nms_radius=nms_radius,
        batch_size=batch_size,
    )
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=device, logger=False, enable_checkpointing=False,
                         enable_progress_bar=progress_bar)
    results = trainer.test(model, dataloaders=test_loader)
    return results[0] if results else {}


def evaluate_objectness_checkpoint(clusters, noise_level, checkpoint, min_sources=1, max_sources=5,
                                   presence_threshold=0.5, pe="none", dropout_num=0, batch_size=512,
                                   workers=4, progress_bar=True, test_size=100, val_size=50):
    """
    Re-evaluate an already-trained LitObjectnessNSource checkpoint on a fresh test-set draw --
    no retraining. Twin of evaluate_checkpoint above (which only covers LitThresholdBaseline);
    needed here so both methods can be scored under the same (e.g. new-metric) test conditions
    for a direct comparison. Same fresh-random-test-set caveat applies -- see that docstring.
    """
    leadfield, noise_dir = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    num_dipoles = leadfield.shape[1]
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    _, test_noise = getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True)
    test_dataset = VariableNSourceDataset(num_dipoles, leadfield, test_size, min_sources=min_sources,
                                         max_sources=max_sources, dist_table=dist_table,
                                         noise_vectors=test_noise)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    model = LitObjectnessNSource.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        max_sources=max_sources, cluster_locs=cluster_locs, dist_table=dist_table, pe=pe,
        dropout_num=dropout_num, presence_threshold=presence_threshold, batch_size=batch_size,
    )
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=device, logger=False, enable_checkpointing=False,
                         enable_progress_bar=progress_bar)
    results = trainer.test(model, dataloaders=test_loader)
    return results[0] if results else {}


def _print_comparison(results, min_sources, max_sources):
    metrics = ["test_loss", "test_precision", "test_recall", "test_hit_rate", "test_f1",
              "test_exact_match", "test_count_mae", "test_count_acc", "test_euclid_dist",
              "test_geodist", "test_chamfer_p2t_euclid", "test_chamfer_t2p_euclid",
              "test_chamfer_p2t_geo", "test_chamfer_t2p_geo"]
    print(f"\n=== Unknown-count ablation summary (n ~ Uniform[{min_sources}, {max_sources}]) ===")
    header = "{:<28}".format("metric") + "".join("{:<26}".format(name) for name in results.keys())
    print(header)
    for metric in metrics:
        row = "{:<28}".format(metric)
        for name in results:
            val = results[name].get(metric)
            row += "{:<26}".format("n/a" if val is None else f"{val:.4f}")
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG unknown-source-count ablation: threshold-decoded "
                                                 "heatmap vs. objectness-gated multi-embedding")
    parser.add_argument("clusters", type=str, help='Cluster resolution (e.g., "10mm", "5mm")')
    parser.add_argument("noise_level", type=float, help="Noise level (e.g., 0.50)")
    parser.add_argument("--min_sources", type=int, default=1, help="Minimum simultaneous sources per sample")
    parser.add_argument("--max_sources", type=int, default=5, help="Maximum simultaneous sources per sample")
    parser.add_argument("--method", type=str, default="both", choices=["topk", "objectness", "both"],
                        help="Which model(s) to train: the threshold-decoded heatmap baseline, "
                             "our objectness-gated multi-embedding method, or both (default)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--scheduler", type=str, default="onecycle")
    parser.add_argument("--pe", type=str, default="none", help="Positional encoding: none, enc, dec, both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.add_argument("--min_dist", type=float, default=20.0,
                        help="Minimum pairwise geodesic distance (mm) between sources")
    parser.add_argument("--prob_threshold", type=float, default=0.5,
                        help="Heatmap baseline: sigmoid probability threshold for count-free decoding")
    parser.add_argument("--presence_threshold", type=float, default=0.5,
                        help="Objectness model: presence probability threshold for count-free decoding")
    parser.add_argument("--pos_weight_scale", type=float, default=1.0,
                        help="Multiplier on the heatmap baseline's BCE positive-class weight")
    parser.add_argument("--loc_weight", type=float, default=1.0,
                        help="Weight on the location-classification term of the objectness matching loss")
    parser.add_argument("--presence_weight", type=float, default=1.0,
                        help="Weight on the presence/objectness term of the objectness matching loss")
    parser.add_argument("--dropout_num", type=int, default=0,
                        help="Number of electrodes to randomly drop per sample during training")
    parser.add_argument("--nms_radius", type=float, default=20.0,
                        help="Geodesic suppression radius (mm) for the threshold baseline's eval-time "
                             "non-maximum-suppression decode; 0 disables NMS (plain thresholding). "
                             "Default matches --min_dist.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Log with TensorBoard instead of the default, Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (default: eeg-ablation-unknown-count)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team (default: your default W&B entity)")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Skip training entirely and just re-evaluate this .ckpt (no retraining). "
                             "Which class it's loaded as is set by --eval_model.")
    parser.add_argument("--eval_model", type=str, default="threshold", choices=["threshold", "objectness"],
                        help="Which model class --eval_checkpoint is a checkpoint for (default: threshold)")
    args = parser.parse_args()

    if args.eval_checkpoint:
        if args.eval_model == "objectness":
            result = evaluate_objectness_checkpoint(
                clusters=args.clusters,
                noise_level=args.noise_level,
                checkpoint=args.eval_checkpoint,
                min_sources=args.min_sources,
                max_sources=args.max_sources,
                presence_threshold=args.presence_threshold,
                pe=args.pe,
                dropout_num=args.dropout_num,
                batch_size=args.batch_size,
                workers=args.workers,
                progress_bar=args.progress_bar,
            )
        else:
            result = evaluate_checkpoint(
                clusters=args.clusters,
                noise_level=args.noise_level,
                checkpoint=args.eval_checkpoint,
                min_sources=args.min_sources,
                max_sources=args.max_sources,
                nms_radius=args.nms_radius,
                pe=args.pe,
                dropout_num=args.dropout_num,
                batch_size=args.batch_size,
                workers=args.workers,
                progress_bar=args.progress_bar,
            )
        print(f"\n=== Re-evaluation ({args.eval_model}) ===")
        for k, v in result.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        run_ablation(
            clusters=args.clusters,
            noise_level=args.noise_level,
            min_sources=args.min_sources,
            max_sources=args.max_sources,
            method=args.method,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            scheduler=args.scheduler,
            pe=args.pe,
            workers=args.workers,
            progress_bar=args.progress_bar,
            min_dist=args.min_dist,
            prob_threshold=args.prob_threshold,
            presence_threshold=args.presence_threshold,
            pos_weight_scale=args.pos_weight_scale,
            loc_weight=args.loc_weight,
            presence_weight=args.presence_weight,
            dropout_num=args.dropout_num,
            nms_radius=args.nms_radius,
            use_tensorboard=args.tensorboard,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
