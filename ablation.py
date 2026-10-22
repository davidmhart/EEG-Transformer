"""
ablation.py -- baseline comparison: "single heatmap + top-k" vs. multi-embedding source localization.

WHAT THIS ABLATES
-----------------
Our method (clusterNetworks.TransformerEncoder with num_sources > 1) gives every simultaneous
source its own learned query embedding. Each embedding independently emits a softmax
distribution over the C candidate patch locations, so an n-source problem is solved as n
independent (but jointly-trained, permutation-matched) classification problems.

The dominant alternative in the EEG/MEG source-imaging literature instead uses a *single*
output layer of width C: one scalar per candidate dipole/patch representing the
probability (or estimated strength) that a source is active there. The n active locations
are then read off by taking the top-k entries of that single vector. This is the
parameterization used by, e.g., ConvDip (Hecker et al., 2021, Frontiers in Neuroscience;
also see the 2020 bioRxiv preprint) and DeepSIF (Sun et al., 2022, Nature Communications;
and its interictal-spike/HFO follow-ups), and by earlier dipole-counting CNN/MLP work such as
Cui et al. (2019, IEEE EMBC) and the Edge Sparse Basis Network (arXiv:2102.09188). None of
that prior work needs a permutation-invariant matching loss, because there is only one output
vector -- "which of the n slots is source 1 vs source 2" is not a question that parameterization
ever has to answer. See the module-level docstring further down (`_permutation_invariant_ce`)
for why our parameterization does need one, and how it generalizes the paper's hand-written
2-source case (main.py's LitNetwork._pi_loss) to arbitrary n via optimal bipartite matching.

WHAT THIS SCRIPT DOES
----------------------
For a chosen n (`--num_sources`), it trains and evaluates two models on the *same* n-source
mixture data (see `NSourceDataset`, which generalizes myDatasets.MultiClusterDataset from
exactly 2 sources to arbitrary n):

  1. `TopKHeatmapNetwork` / `LitTopKBaseline`  -- the standard single-heatmap approach.
     Single dense "heatmap" head over all C locations, trained with (weighted) multi-label
     BCE, decoded via top-n.
  2. `LitMultiEmbedNSource` -- our method, but with the loss generalized from 2 to n sources
     so the comparison is fair at any n.

Both models share the *same* electrode-tokenizing transformer backbone
(clusterNetworks.TransformerEncoder's tokenization) -- only the output head and loss differ --
so the ablation isolates the representation choice, not encoder capacity.

Usage:
    python ablation.py 10mm 0.50 --num_sources 3 --method both --max_epochs 200

References
----------
- Hecker, L., Rupprecht, R., Elst, L.T., Kornmeier, J. (2021). ConvDip: A Convolutional
  Neural Network for Better EEG Source Imaging. Frontiers in Neuroscience, 15:569918.
  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.569918/full
- Sun, R., Sohrabpour, A., Worrell, G.A., He, B. (2022). Deep neural networks constrained by
  neural mass models improve electrophysiological source imaging of spatiotemporal brain
  dynamics (DeepSIF). PNAS / Nat. Commun. family of papers.
- Cui, S. et al. (2019). EEG Source Localization Using Spatio-Temporal Neural Network.
  IEEE EMBC.
- Wei, X. et al. (2021). Edge Sparse Basis Network: A Deep Learning Framework for EEG Source
  Localization. arXiv:2102.09188.
- Carion, N. et al. (2020). End-to-End Object Detection with Transformers (DETR).
  arXiv:2005.12872. -- source of the bipartite/Hungarian set-matching loss idea this script's
  `_permutation_invariant_ce` generalizes our 2-source loss with.
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. arXiv:1708.02002. --
  motivates the `pos_weight`-balanced BCE used for the sparse multi-label heatmap baseline.
"""

import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from helpers import load_data, getNoiseVectors, load_dist_table, load_locs, load_electrode_locs
from clusterNetworks import TransformerEncoder, _drop_electrode_tokens
from positional_encoding_3d import PositionalEncoding3D


# ============================================================================
# Data: general n-source mixture (generalizes myDatasets.MultiClusterDataset)
# ============================================================================

class NSourceDataset(Dataset):
    """
    General n-source generalization of MultiClusterDataset (myDatasets.py).

    Each sample mixes `num_sources` distinct, sufficiently-separated leadfield columns with
    independent relative-strength weights w_i ~ Uniform(alpha_min, alpha_max), rescaled so the
    strongest source has unit amplitude:
        signal = (1 / max_i w_i) * sum_i w_i * leadfield[:, cluster_i]  + noise
    At num_sources=2 this reduces exactly to MultiClusterDataset's alpha/(1-alpha) mixing.

    Returns (signal, clusters, weights) where clusters is a (num_sources,) long tensor of
    location indices and weights is a (num_sources,) float32 tensor of relative source
    strengths normalized to sum to 1 (usable as a soft "strength" target by the baseline).
    """

    def __init__(self, num_dipoles, leadfield, num_examples, num_sources=2, dist_table=None,
                 min_dist=20.0, noise_vectors=None, noise_level=0.0, noise_mode="sensor",
                 alpha_min=0.25, alpha_max=0.75):
        super().__init__()
        if num_sources < 2:
            raise ValueError("NSourceDataset requires num_sources >= 2")
        self.num_dipoles = num_dipoles
        self.leadfield = leadfield
        self.num_examples = num_examples
        self.num_sources = num_sources
        self.dist_table = dist_table
        self.min_dist = min_dist if dist_table is not None else 0.0
        self.noise_mode = noise_mode
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.length = num_examples * num_dipoles
        self._pairs = list(itertools.combinations(range(num_sources), 2))

        if noise_vectors is None:
            self.noise_vectors = None
            noise_max = np.amax(np.abs(leadfield), axis=0)
            self.std_devs = noise_max / 2 * noise_level
        else:
            self.noise_vectors = torch.tensor(noise_vectors, dtype=torch.float32)

    def _sample_clusters(self):
        clusters = None
        for _ in range(1000):
            candidate = np.random.choice(self.num_dipoles, self.num_sources, replace=False)
            if self.dist_table is None:
                return candidate
            if all(self.dist_table[candidate[i], candidate[j]] >= self.min_dist for i, j in self._pairs):
                return candidate
            clusters = candidate
        return clusters  # best-effort fallback if no valid draw found in 1000 tries

    def __getitem__(self, idx):
        example_idx = idx // self.num_dipoles
        clusters = self._sample_clusters()
        n = self.num_sources

        weights = np.random.uniform(self.alpha_min, self.alpha_max, size=n).astype(np.float32)
        scale = 1.0 / max(weights.max(), 1e-8)

        lf = torch.zeros(self.leadfield.shape[0], dtype=torch.float32)
        for i in range(n):
            lf_i = torch.tensor(self.leadfield[:, int(clusters[i])], dtype=torch.float32)
            lf = lf + (scale * float(weights[i])) * lf_i

        c0 = int(clusters[0])
        if self.noise_vectors is None:
            if self.noise_mode == "per_source":
                noise = torch.zeros_like(lf)
                for c in clusters:
                    noise = noise + torch.normal(0.0, float(self.std_devs[int(c)]), size=lf.shape)
            else:
                noise = torch.normal(0.0, float(self.std_devs[c0]), size=lf.shape)
        else:
            ni = example_idx % self.noise_vectors.shape[0]
            if self.noise_mode == "per_source":
                noise = torch.zeros_like(lf)
                for c in clusters:
                    noise = noise + self.noise_vectors[ni, :, int(c)]
            else:
                noise = self.noise_vectors[ni, :, c0]

        return (lf + noise,
                torch.tensor(clusters, dtype=torch.long),
                torch.tensor(weights / weights.sum(), dtype=torch.float32))

    def __len__(self):
        return self.length


# ============================================================================
# Baseline network: single dense "heatmap" head over all C candidate locations
# ============================================================================

class TopKHeatmapNetwork(nn.Module):
    """
    Standard-approach output head: one learned pooling token (like a ViT/BERT [CLS] token)
    reads the electrode tokens and is projected to a length-C vector of location logits.
    Trained as multi-label classification (or soft-strength regression); the top-n entries
    at inference time are the predicted source locations.

    Uses the identical electrode tokenization (raw or 3D-positional-encoded) as
    clusterNetworks.TransformerEncoder so the ablation isolates only the output
    parameterization, not the encoder.
    """

    def __init__(self, in_channels, out_channels, electrode_locs, d_model=512, nhead=8,
                 num_encoder_layers=3, with_pos_enc=False, dropout=0.1, dropout_num=0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

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
        self.readout_token = nn.Parameter(torch.rand(1, d_model))
        self.output_projection = nn.Linear(d_model, out_channels)

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

        token = self.readout_token.unsqueeze(0).expand(x_tokens.size(0), -1, -1)
        x_out = self.transformer_encoder(torch.cat((token, x_tokens), dim=1))
        return self.output_projection(x_out[:, 0, :])  # (batch, out_channels) heatmap logits


# ============================================================================
# Permutation-invariant loss for the n-embedding method, generalized beyond n=2
# ============================================================================

def _permutation_invariant_ce(logits, labels, loss_func, max_bruteforce_n=7):
    """
    Why this is needed (and how it generalizes main.py's `LitNetwork._pi_loss`):

    With n learned query embeddings, embedding slot i's softmax has no fixed target -- slot 0
    might learn to grab whichever ground-truth source is geometrically "first" for one sample
    and the other source for the next. Training against a *fixed* slot-to-label pairing (slot 0
    -> label 0, slot 1 -> label 1, ...) would punish a perfectly correct prediction that simply
    reported the sources in the "wrong" order, which actively fights the model during training.
    The original 2-source code sidesteps this by brute-force comparing both orderings
    (l00+l11 vs l01+l10) and taking the min -- exactly a bipartite matching problem restricted
    to n=2, where there are only 2 possible slot<->label assignments.

    This function is that same idea for general n: build the n x n cost matrix
    cost[i, j] = CE(logits[:, i, :], labels[:, j]), then find the assignment of slots to labels
    that minimizes total cost (Kuhn-Munkres / Hungarian matching -- the same bipartite-matching
    loss used for set prediction in DETR, Carion et al. 2020). For n <= max_bruteforce_n we
    instead enumerate all n! permutations directly (fully vectorized across the batch, no
    per-sample Python loop, and exactly what the existing 2-source code does for n=2); for
    larger n we fall back to scipy's per-sample Hungarian solver.

    logits: (B, n, C) unnormalized per-slot class scores.
    labels: (B, n) ground-truth location index for each of the n sources (order arbitrary).
    loss_func: elementwise CE, i.e. nn.CrossEntropyLoss(reduction="none").

    Returns (loss: scalar mean per-source CE at the optimal matching,
             assignment: (B, n) long tensor, assignment[b, i] = index into labels[b] matched
             to output slot i).
    """
    B, n, C = logits.shape

    if n <= max_bruteforce_n:
        perms = list(itertools.permutations(range(n)))
        costs = torch.empty(len(perms), B, device=logits.device)
        flat_logits = logits.reshape(B * n, C)
        for p, perm in enumerate(perms):
            perm_labels = labels[:, perm].reshape(B * n)
            costs[p] = loss_func(flat_logits, perm_labels).reshape(B, n).sum(dim=1)
        best_cost, best_perm_idx = costs.min(dim=0)
        perms_t = torch.tensor(perms, device=logits.device, dtype=torch.long)
        assignment = perms_t[best_perm_idx]
        return best_cost.mean() / n, assignment

    # Hungarian fallback for large n: build the (B, n, n) cost matrix, solve per-sample.
    logp = torch.log_softmax(logits, dim=-1)  # (B, n, C)
    label_idx = labels.unsqueeze(1).expand(B, n, n)  # (B, n_slot, n_label)
    cost = -torch.gather(logp, dim=2, index=label_idx)  # cost[b, i, j] = CE(slot i, label j)
    cost_np = cost.detach().cpu().numpy()
    assignment = torch.empty(B, n, dtype=torch.long, device=logits.device)
    total = torch.zeros((), device=logits.device)
    for b in range(B):
        row, col = linear_sum_assignment(cost_np[b])
        assignment[b, row] = torch.tensor(col, device=logits.device)
        total = total + cost[b, row, col].sum()
    return total / (B * n), assignment


# ============================================================================
# Shared evaluation: bipartite matching between a predicted location *set* and the
# true location set, usable for either model's output (top-n heatmap or n argmaxed slots)
# ============================================================================

def _bipartite_match_metrics(pred_idx, true_idx, cluster_locs=None, dist_table=None):
    """
    pred_idx, true_idx: (B, n) long tensors of location indices (same n on both sides,
    order arbitrary on both sides -- exactly the situation for a top-n heatmap readout,
    and, for a fair comparison, also how we score our own method's raw per-slot argmax).

    Returns dict with:
      hit_rate    - mean fraction of true locations present verbatim in the predicted set
      exact_match - fraction of samples where predicted set == true set
      euclid_dist - mean optimally-matched Euclidean distance (if cluster_locs given)
      geodist     - mean optimally-matched geodesic distance, ignoring unreachable (inf) pairs
                    (if dist_table given)
    """
    B, n = pred_idx.shape
    pred_np = pred_idx.detach().cpu().numpy()
    true_np = true_idx.detach().cpu().numpy()

    hits = np.array([len(set(pred_np[b].tolist()) & set(true_np[b].tolist())) for b in range(B)]) / n
    exact = np.array([set(pred_np[b].tolist()) == set(true_np[b].tolist()) for b in range(B)], dtype=np.float32)

    out = {"hit_rate": float(hits.mean()), "exact_match": float(exact.mean())}

    if cluster_locs is not None:
        euclid_vals = []
        for b in range(B):
            locs_pred = cluster_locs[pred_np[b]]
            locs_true = cluster_locs[true_np[b]]
            cost = torch.cdist(locs_pred, locs_true).cpu().numpy()
            row, col = linear_sum_assignment(cost)
            euclid_vals.append(cost[row, col].mean())
        out["euclid_dist"] = float(np.mean(euclid_vals))

    if dist_table is not None:
        geo_vals = []
        for b in range(B):
            cost_np = dist_table[np.ix_(pred_np[b], true_np[b])]
            if torch.is_tensor(cost_np):
                cost_np = cost_np.cpu().numpy()
            finite_mask = np.isfinite(cost_np)
            cost_for_match = np.where(finite_mask, cost_np, 1e6)
            row, col = linear_sum_assignment(cost_for_match)
            matched = cost_np[row, col]
            matched = matched[np.isfinite(matched)]
            if matched.size:
                geo_vals.append(matched.mean())
        if geo_vals:
            out["geodist"] = float(np.mean(geo_vals))

    return out


# ============================================================================
# Non-maximum suppression for the single-heatmap baselines
# ============================================================================

def nms_select(scores, dist_matrix, radius, k=None, threshold=None):
    """
    Greedy non-maximum suppression decode for a single dense "heatmap" output, shared by
    ablation.py's LitTopKBaseline (fixed-k decode) and ablation_unknown_count.py's
    LitThresholdBaseline (count-free threshold decode).

    Why this exists: a true source's leadfield varies smoothly across nearby cortical
    patches, so a real source generally makes its immediate neighbors light up somewhat too,
    not just the one "true" location. Plain top-k/thresholding has no way to tell "two
    genuinely distinct sources" apart from "two neighboring points describing the same blob" --
    it'll happily spend two of its k slots (or count two suprathreshold points) on one blob's
    two brightest pixels. That's a decode artifact, not evidence the model failed to find a
    second source, and scoring it as a miss would be unfair to this baseline. This is a
    decode-only, eval-time fix -- it never touches the loss (see the discussion in this
    project's history: NMS's discrete sort/argmax/suppression steps aren't usefully
    differentiable, so it isn't something to backprop through; it's applied only to logits
    already detached from the training graph). This mirrors how the EEG/MEG source-imaging
    literature already handles the analogous problem: discrete sources are extracted from a
    continuous/distributed inverse solution via local-maxima extraction with a magnitude
    cutoff, not by taking the raw top-k of a smooth map.

    scores: (B, C) tensor/array of per-location scores (only relative order matters, so
        logits or probabilities both work).
    dist_matrix: (C, C) array of pairwise distances between candidate locations -- the
        geodesic dist_table already used elsewhere in this codebase for --min_dist, so the
        same physical notion of "separation" applies to both true-source sampling and
        predicted-peak decoding.
    radius: suppression radius. After a peak is selected, every other candidate within
        `radius` of it is removed from further consideration. radius <= 0 disables
        suppression entirely (recovers plain top-k / plain thresholding).
    k: if given, always return exactly k selections per sample (used by the fixed-count
        decode). If suppression would leave fewer than k eligible candidates, suppression is
        relaxed just enough to still reach k, so downstream code that expects a fixed-size
        prediction set keeps working unchanged.
    threshold: if given, only candidates with score > threshold are ever eligible, and the
        number of selections is however many distinct peaks survive suppression (used by the
        count-free decode). k and threshold can both be set (cap a threshold-filtered pool at
        k peaks); at least one should normally be given.

    Returns a list of length B of 1D numpy arrays of selected indices, sorted by descending
    score (each has length k if k is given and enough candidates exist, otherwise a variable
    length that may be zero).
    """
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    dist_np = dist_matrix.detach().cpu().numpy() if torch.is_tensor(dist_matrix) else np.asarray(dist_matrix)
    B, C = scores_np.shape
    results = []
    for b in range(B):
        s = scores_np[b]
        order = np.argsort(-s)
        if threshold is not None:
            order = order[s[order] > threshold]

        suppressed = np.zeros(C, dtype=bool)
        selected = []
        if radius > 0:
            for idx in order:
                if k is not None and len(selected) >= k:
                    break
                if suppressed[idx]:
                    continue
                selected.append(int(idx))
                suppressed |= dist_np[idx] <= radius
        elif k is not None:
            selected = order[:k].tolist()
        else:
            selected = order.tolist()

        if k is not None and len(selected) < k:
            # Suppression left too few candidates (e.g. radius large relative to head size) --
            # relax it just enough to still return exactly k, preferring the highest scores.
            selected_set = set(selected)
            for idx in order:
                if len(selected) >= k:
                    break
                if int(idx) not in selected_set:
                    selected.append(int(idx))
                    selected_set.add(int(idx))

        results.append(np.array(selected, dtype=np.int64))
    return results


# ============================================================================
# Shared LightningModule plumbing (optimizer/buffers identical to main.py's LitNetwork)
# ============================================================================

class _BaseAblationModule(pl.LightningModule):
    def __init__(self, batch_size, lr, max_epochs, cluster_locs, dist_table, electrode_locs, scheduler):
        super().__init__()
        self.b = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.scheduler = scheduler

        if cluster_locs is not None:
            self.register_buffer("cluster_locs", torch.tensor(cluster_locs, dtype=torch.float32))
        else:
            self.cluster_locs = None

        if dist_table is not None:
            self.register_buffer("dist_table", torch.tensor(dist_table, dtype=torch.float32))
        else:
            self.dist_table = None

        if electrode_locs is not None:
            self.register_buffer("electrode_locs", torch.tensor(electrode_locs, dtype=torch.float32))
        else:
            self.electrode_locs = None

    def configure_optimizers(self):
        if self.scheduler == "step":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(.75 * self.max_epochs), gamma=0.1)
        elif self.scheduler == "onecycle":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=1, epochs=self.max_epochs)
        else:
            raise ValueError("Scheduler type not recognized")
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def _log_match_metrics(self, split, pred_idx, true_idx):
        metrics = _bipartite_match_metrics(pred_idx, true_idx, self.cluster_locs, self.dist_table)
        for name, value in metrics.items():
            self.log(f"{split}_{name}", value, batch_size=self.b, prog_bar=(name in ("hit_rate", "euclid_dist")),
                      on_step=False, on_epoch=True, sync_dist=True)


# ============================================================================
# Baseline LightningModule: multi-label heatmap + top-k decoding
# ============================================================================

class LitTopKBaseline(_BaseAblationModule):
    def __init__(self, in_channels, out_channels, electrode_locs, num_sources=2, batch_size=512,
                 lr=1e-4, max_epochs=200, cluster_locs=None, dist_table=None, scheduler="onecycle",
                 pe="none", target_mode="binary", pos_weight_scale=1.0, dropout_num=0, nms_radius=20.0):
        super().__init__(batch_size, lr, max_epochs, cluster_locs, dist_table, electrode_locs, scheduler)
        self.num_sources = num_sources
        self.out_channels = out_channels
        self.target_mode = target_mode
        # Suppression radius (mm, geodesic) for the eval-time NMS decode below; 0 recovers
        # plain top-k. See nms_select's docstring for why this only affects decoding, not loss.
        self.nms_radius = nms_radius

        with_pos_enc = pe in ("enc", "both")
        self.model = TopKHeatmapNetwork(in_channels, out_channels, electrode_locs,
                                        with_pos_enc=with_pos_enc, dropout_num=dropout_num)

        # Sparse multi-label targets (n positives out of C candidates): balance the loss the
        # way Focal Loss / weighted-BCE approaches do (Lin et al., 2017), since plain BCE is
        # dominated by the (C - n) negatives once C is in the hundreds/thousands.
        pos_weight = torch.tensor((out_channels - num_sources) / max(num_sources, 1) * pos_weight_scale)
        self.register_buffer("pos_weight", pos_weight)
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def _make_targets(self, clusters, weights):
        target = torch.zeros(clusters.shape[0], self.out_channels, device=clusters.device)
        if self.target_mode == "strength":
            target.scatter_(1, clusters, weights)
        else:
            target.scatter_(1, clusters, 1.0)
        return target

    def _step(self, data, split):
        leadfield, clusters, weights = data[0], data[1], data[2]
        logits = self.model(leadfield)
        target = self._make_targets(clusters, weights)
        loss = self.loss_func(logits, target)
        self.log(f"{split}_loss", loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if split != "train":
            probs = torch.sigmoid(logits)
            pred_idx_list = nms_select(probs, self.dist_table, self.nms_radius, k=self.num_sources)
            pred_idx = torch.tensor(np.stack(pred_idx_list), device=logits.device)
            self._log_match_metrics(split, pred_idx, clusters)

        return loss

    def training_step(self, data, batch_idx):
        return self._step(data, "train")

    def validation_step(self, data, batch_idx):
        return self._step(data, "val")

    def test_step(self, data, batch_idx):
        return self._step(data, "test")


# ============================================================================
# Our method's LightningModule, generalized to n sources
# ============================================================================

class LitMultiEmbedNSource(_BaseAblationModule):
    def __init__(self, in_channels, out_channels, electrode_locs, num_sources=2, batch_size=512,
                 lr=1e-4, max_epochs=200, cluster_locs=None, dist_table=None, scheduler="onecycle",
                 pe="none", dropout_num=0):
        super().__init__(batch_size, lr, max_epochs, cluster_locs, dist_table, electrode_locs, scheduler)
        self.num_sources = num_sources

        with_pos_enc = pe in ("enc", "both")
        # Same backbone as the paper's real model (main.py's LitNetwork uses this exact class);
        # only the loss is generalized here so it isn't limited to num_sources==2.
        self.model = TransformerEncoder(in_channels, out_channels, electrode_locs,
                                        num_sources=num_sources, with_pos_enc=with_pos_enc,
                                        dropout_num=dropout_num)
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def _step(self, data, split):
        leadfield, clusters, _weights = data[0], data[1], data[2]
        logits = self.model(leadfield)  # (B, num_sources, C)
        loss, _assignment = _permutation_invariant_ce(logits, clusters, self.loss_func)
        self.log(f"{split}_loss", loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if split != "train":
            pred_idx = torch.argmax(logits, dim=-1)  # (B, num_sources), raw per-slot predictions
            self._log_match_metrics(split, pred_idx, clusters)

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

def _build_loaders(clusters, noise_level, num_sources, min_dist, alpha_min, alpha_max,
                   batch_size, workers, train_size=100, val_size=50, test_size=100):
    leadfield, noise_dir = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    num_dipoles = leadfield.shape[1]

    val_noise, test_noise = getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True)

    train_dataset = NSourceDataset(num_dipoles, leadfield, train_size, num_sources=num_sources,
                                   dist_table=dist_table, min_dist=min_dist, noise_level=noise_level,
                                   alpha_min=alpha_min, alpha_max=alpha_max)
    val_dataset = NSourceDataset(num_dipoles, leadfield, val_size, num_sources=num_sources,
                                 dist_table=dist_table, min_dist=min_dist, noise_vectors=val_noise,
                                 alpha_min=alpha_min, alpha_max=alpha_max)
    test_dataset = NSourceDataset(num_dipoles, leadfield, test_size, num_sources=num_sources,
                                  dist_table=dist_table, min_dist=min_dist, noise_vectors=test_noise,
                                  alpha_min=alpha_min, alpha_max=alpha_max)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    return (train_loader, val_loader, test_loader, leadfield, cluster_locs, dist_table, electrode_locs)


def _make_logger(use_tensorboard, log_dir, log_name, wandb_project, wandb_entity, group, tags):
    if use_tensorboard:
        return pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name)
    return pl_loggers.WandbLogger(project=wandb_project, entity=wandb_entity, name=log_name,
                                  save_dir=log_dir, group=group, tags=tags)


def _train_and_test(model, train_loader, val_loader, test_loader, max_epochs, progress_bar, log_name, log_dir,
                    use_tensorboard=False, wandb_project=None, wandb_entity=None, group=None, tags=None):
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_hit_rate", save_top_k=1, mode="max")
    logger = _make_logger(use_tensorboard, log_dir, log_name, wandb_project, wandb_entity, group, tags)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=[checkpoint],
                         logger=logger, enable_progress_bar=progress_bar, check_val_every_n_epoch=10, log_every_n_steps=10)
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    if not use_tensorboard:
        import wandb  # only needed (and only imported) when W&B logging is in use
        wandb.finish()  # close this run so the next model in the same process starts a fresh one
    return results[0] if results else {}


def run_ablation(clusters, noise_level, num_sources=2, method="both", batch_size=512, max_epochs=200,
                 learning_rate=1e-4, scheduler="onecycle", pe="none", workers=4, progress_bar=True,
                 min_dist=20.0, alpha_min=0.25, alpha_max=0.75, target_mode="binary",
                 pos_weight_scale=1.0, dropout_num=0, nms_radius=20.0, use_tensorboard=False,
                 wandb_project=None, wandb_entity=None):

    (train_loader, val_loader, test_loader, leadfield,
     cluster_locs, dist_table, electrode_locs) = _build_loaders(
        clusters, noise_level, num_sources, min_dist, alpha_min, alpha_max, batch_size, workers)

    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]
    s = "{s:.2f}".format(s=noise_level).split(".")[1]
    log_dir = f"my_logs/ablation_cluster{clusters}_{num_sources}src/noise_{s}"
    group = f"{clusters}_{num_sources}src_noise{s}"
    wandb_project = wandb_project or "eeg-ablation"

    results = {}

    if method in ("topk", "both"):
        topk_model = LitTopKBaseline(in_channels, out_channels, electrode_locs, num_sources=num_sources,
                                     batch_size=batch_size, lr=learning_rate, max_epochs=max_epochs,
                                     cluster_locs=cluster_locs, dist_table=dist_table, scheduler=scheduler,
                                     pe=pe, target_mode=target_mode, pos_weight_scale=pos_weight_scale,
                                     dropout_num=dropout_num, nms_radius=nms_radius)
        log_name = f"TopKHeatmap_{target_mode}_{scheduler}_{max_epochs}_{learning_rate:.0e}"
        results["topk_heatmap_baseline"] = _train_and_test(
            topk_model, train_loader, val_loader, test_loader, max_epochs, progress_bar, log_name, log_dir,
            use_tensorboard=use_tensorboard, wandb_project=wandb_project, wandb_entity=wandb_entity,
            group=group, tags=["topk_heatmap_baseline"])

    if method in ("multiembed", "both"):
        multi_model = LitMultiEmbedNSource(in_channels, out_channels, electrode_locs, num_sources=num_sources,
                                           batch_size=batch_size, lr=learning_rate, max_epochs=max_epochs,
                                           cluster_locs=cluster_locs, dist_table=dist_table, scheduler=scheduler,
                                           pe=pe, dropout_num=dropout_num)
        log_name = f"MultiEmbed_{scheduler}_{max_epochs}_{learning_rate:.0e}"
        results["multi_embedding_ours"] = _train_and_test(
            multi_model, train_loader, val_loader, test_loader, max_epochs, progress_bar, log_name, log_dir,
            use_tensorboard=use_tensorboard, wandb_project=wandb_project, wandb_entity=wandb_entity,
            group=group, tags=["multi_embedding_ours"])

    if results:
        _print_comparison(results, num_sources)

    return results


def evaluate_checkpoint(clusters, noise_level, checkpoint, num_sources=2, nms_radius=20.0,
                        pe="none", dropout_num=0, batch_size=512, workers=4, progress_bar=True,
                        test_size=100, val_size=50):
    """
    Re-evaluate an already-trained LitTopKBaseline checkpoint on a fresh test-set draw with a
    chosen nms_radius -- no retraining. Since nms_select only affects eval-time decoding (see
    its docstring), this is exactly how you compare decode strategies (e.g. nms_radius=0, the
    original plain top-k, vs. nms_radius=20, the NMS decode) on identical trained weights.

    Note: this test set is a fresh random draw (only the additive noise is cached/reproducible
    -- see helpers.getNoiseVectors -- the source locations themselves are re-sampled), so the
    numbers are a new, independent sample from the same distribution the model was originally
    tested on, not a literal replay of the original trainer.test() call.
    """
    leadfield, noise_dir = load_data(clusters)
    cluster_locs, _ = load_locs(clusters)
    dist_table = load_dist_table(clusters)
    electrode_locs = load_electrode_locs()
    num_dipoles = leadfield.shape[1]
    in_channels, out_channels = leadfield.shape[0], leadfield.shape[1]

    _, test_noise = getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True)
    test_dataset = NSourceDataset(num_dipoles, leadfield, test_size, num_sources=num_sources,
                                  dist_table=dist_table, noise_vectors=test_noise)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    model = LitTopKBaseline.load_from_checkpoint(
        checkpoint, map_location="cpu",
        in_channels=in_channels, out_channels=out_channels, electrode_locs=electrode_locs,
        num_sources=num_sources, cluster_locs=cluster_locs, dist_table=dist_table,
        pe=pe, dropout_num=dropout_num, nms_radius=nms_radius, batch_size=batch_size,
    )
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=device, logger=False, enable_checkpointing=False,
                         enable_progress_bar=progress_bar)
    results = trainer.test(model, dataloaders=test_loader)
    return results[0] if results else {}


def _print_comparison(results, num_sources):
    metrics = ["test_loss", "test_hit_rate", "test_exact_match", "test_euclid_dist", "test_geodist"]
    print(f"\n=== Ablation summary (num_sources={num_sources}) ===")
    header = "{:<28}".format("metric") + "".join("{:<22}".format(name) for name in results.keys())
    print(header)
    for metric in metrics:
        row = "{:<28}".format(metric)
        for name in results:
            val = results[name].get(metric)
            row += "{:<22}".format("n/a" if val is None else f"{val:.4f}")
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG multi-source ablation: heatmap+top-k vs. multi-embedding")
    parser.add_argument("clusters", type=str, help='Cluster resolution (e.g., "10mm", "5mm")')
    parser.add_argument("noise_level", type=float, help="Noise level (e.g., 0.50)")
    parser.add_argument("--num_sources", type=int, default=2, help="Number of simultaneous sources (n >= 2)")
    parser.add_argument("--method", type=str, default="both", choices=["topk", "multiembed", "both"],
                        help="Which model(s) to train: the top-k heatmap baseline, our multi-embedding "
                             "method, or both (default) for a direct comparison")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--scheduler", type=str, default="onecycle")
    parser.add_argument("--pe", type=str, default="none", help="Positional encoding: none, enc, dec, both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.add_argument("--min_dist", type=float, default=20.0,
                        help="Minimum pairwise geodesic distance (mm) between sources")
    parser.add_argument("--alpha_range", type=float, nargs=2, default=[0.25, 0.75], metavar=("MIN", "MAX"),
                        help="Per-source relative-strength sampling range (default: 0.25 0.75)")
    parser.add_argument("--target_mode", type=str, default="binary", choices=["binary", "strength"],
                        help="Baseline heatmap target: binary presence (default) or continuous "
                             "relative-strength regression")
    parser.add_argument("--pos_weight_scale", type=float, default=1.0,
                        help="Multiplier on the baseline's BCE positive-class weight (default: 1.0, "
                             "i.e. exactly balance the (C - n) negatives against the n positives)")
    parser.add_argument("--dropout_num", type=int, default=0,
                        help="Number of electrodes to randomly drop per sample during training")
    parser.add_argument("--nms_radius", type=float, default=20.0,
                        help="Geodesic suppression radius (mm) for the topk baseline's eval-time "
                             "non-maximum-suppression decode; 0 disables NMS (plain top-k). "
                             "Default matches --min_dist.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Log with TensorBoard instead of the default, Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (default: eeg-ablation)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team (default: your default W&B entity)")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Skip training entirely and just re-evaluate this LitTopKBaseline "
                             ".ckpt with the given --nms_radius (no retraining).")
    args = parser.parse_args()

    if args.eval_checkpoint:
        result = evaluate_checkpoint(
            clusters=args.clusters,
            noise_level=args.noise_level,
            checkpoint=args.eval_checkpoint,
            num_sources=args.num_sources,
            nms_radius=args.nms_radius,
            pe=args.pe,
            dropout_num=args.dropout_num,
            batch_size=args.batch_size,
            workers=args.workers,
            progress_bar=args.progress_bar,
        )
        print(f"\n=== Re-evaluation (nms_radius={args.nms_radius}) ===")
        for k, v in result.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        run_ablation(
            clusters=args.clusters,
            noise_level=args.noise_level,
            num_sources=args.num_sources,
            method=args.method,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            scheduler=args.scheduler,
            pe=args.pe,
            workers=args.workers,
            progress_bar=args.progress_bar,
            min_dist=args.min_dist,
            alpha_min=args.alpha_range[0],
            alpha_max=args.alpha_range[1],
            target_mode=args.target_mode,
            pos_weight_scale=args.pos_weight_scale,
            dropout_num=args.dropout_num,
            nms_radius=args.nms_radius,
            use_tensorboard=args.tensorboard,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
