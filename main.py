import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader

from helpers import load_data, getNoiseVectors, load_dist_table, load_locs, load_electrode_locs
from myDatasets import ClusterDataset, MultiClusterDataset
from clusterNetworks import LinearNetwork, TransformerEncoder

import numpy as np
import os
from scipy.io import loadmat, savemat


class LitNetwork(pl.LightningModule):
    def __init__(self, in_channels, out_channels, network_name="Linear", batch_size=1, lr=1e-4,
                 max_epochs=200, cluster_locs=None, dist_table=None, electrode_locs=None,
                 scheduler="onecycle", pe="none", num_sources=1,
                 predict_alpha=False, alpha_weight=0.1, dropout_num=0):
        super(LitNetwork, self).__init__()

        self.num_sources = num_sources
        self.predict_alpha = bool(predict_alpha)
        self.alpha_weight = float(alpha_weight)

        if network_name == "Linear":
            self.model = LinearNetwork(in_channels, out_channels, dropout_num=dropout_num)
        elif network_name == "TransformerEncoder":
            with_pos_enc = pe in ("enc", "both")
            self.model = TransformerEncoder(in_channels, out_channels, electrode_locs,
                                            num_sources=num_sources, with_pos_enc=with_pos_enc,
                                            dropout_num=dropout_num)
        else:
            raise ValueError(f"Network name not recognized: {network_name}")

        # Multi-source uses per-element CE so permutation-invariant matching works;
        # single-source uses the standard mean-reduced CE.
        if num_sources > 1:
            self.loss_func = nn.CrossEntropyLoss(reduction="none")
            if self.predict_alpha:
                self.alpha_head = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
                self.alpha_loss_fn = nn.L1Loss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

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

        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=out_channels, average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=out_channels, average='micro')

    # ------------------------------------------------------------------
    # Permutation-invariant loss for two-source prediction
    # out: (batch, 2, num_clusters)   clusters: (batch, 2)
    # Returns scalar loss and a swap_mask bool tensor (batch,)
    # ------------------------------------------------------------------
    def _pi_loss(self, out, clusters):
        logits0, logits1 = out[:, 0, :], out[:, 1, :]
        labels0, labels1 = clusters[:, 0], clusters[:, 1]

        l00 = self.loss_func(logits0, labels0)
        l11 = self.loss_func(logits1, labels1)
        l01 = self.loss_func(logits0, labels1)
        l10 = self.loss_func(logits1, labels0)

        loss_nat = l00 + l11
        loss_swap = l01 + l10
        swap_mask = loss_swap < loss_nat
        return torch.min(loss_nat, loss_swap).mean(), swap_mask

    def _strength_metrics(self, pred0_matched, pred1_matched, labels0, labels1, alpha_true):
        """
        Compute strong-source accuracy, weak-source accuracy, and alpha-weighted accuracy.
        pred0_matched / pred1_matched are already permutation-corrected so that
        pred0_matched corresponds to labels0 (the source with weight alpha_true).
        """
        alpha = alpha_true.view(-1).detach()
        correct0 = (pred0_matched == labels0).float()
        correct1 = (pred1_matched == labels1).float()

        weighted_acc = (alpha * correct0 + (1.0 - alpha) * correct1).mean()

        strong_is_0 = alpha >= 0.5
        strong_acc = torch.where(strong_is_0, correct0, correct1).mean()
        weak_acc = torch.where(strong_is_0, correct1, correct0).mean()

        return strong_acc, weak_acc, weighted_acc

    def forward(self, x):
        class_logits = self.model(x)
        if self.predict_alpha and self.num_sources > 1:
            # Pool source logits to a single vector, then predict scalar alpha
            alpha_pred = self.alpha_head(class_logits.mean(dim=1)).squeeze(-1)
            return class_logits, alpha_pred
        return class_logits

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

    # ------------------------------------------------------------------
    # Single-source steps
    # ------------------------------------------------------------------
    def training_step(self, data, batch_idx):
        if self.num_sources > 1:
            return self._training_step_multi(data, batch_idx)

        leadfield, cluster = data[0], data[1]
        out = self.forward(leadfield)
        loss = self.loss_func(out, cluster)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        return loss

    def validation_step(self, val_data, batch_idx):
        if self.num_sources > 1:
            return self._validation_step_multi(val_data, batch_idx)

        leadfield, cluster = val_data[0], val_data[1]
        out = self.forward(leadfield)
        loss = self.loss_func(out, cluster)

        self.log("val_loss", loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_acc(out, cluster)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.cluster_locs is not None:
            pred_cluster = torch.argmax(out, dim=1)
            euclid_dist = torch.norm(self.cluster_locs[pred_cluster] - self.cluster_locs[cluster], dim=1)
            self.log("val_euclid_dist", torch.mean(euclid_dist), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.dist_table is not None:
            indices = torch.argmax(out, dim=1)
            geodist = self.dist_table[indices, cluster]
            num_inf = torch.sum(torch.isinf(geodist)).to(torch.float32)
            geodist = geodist[~torch.isinf(geodist)]
            self.log("val_geodist", torch.mean(geodist), batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_geodist_perc_inf", num_inf / self.b, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None

    def test_step(self, test_data, batch_idx):
        if self.num_sources > 1:
            return self._test_step_multi(test_data, batch_idx)

        leadfield, cluster = test_data[0], test_data[1]
        out = self.forward(leadfield)
        loss = self.loss_func(out, cluster)

        self.log("test_loss", loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_acc(out, cluster)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("hp_metric", self.test_acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        if self.cluster_locs is not None:
            pred_cluster = torch.argmax(out, dim=1)
            euclid_dist = torch.norm(self.cluster_locs[pred_cluster] - self.cluster_locs[cluster], dim=1)
            self.log("test_euclid_dist", torch.mean(euclid_dist), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.dist_table is not None:
            indices = torch.argmax(out, dim=1)
            geodist = self.dist_table[indices, cluster]
            num_inf = torch.sum(torch.isinf(geodist)).to(torch.float32)
            geodist = geodist[~torch.isinf(geodist)]
            self.log("test_geodist", torch.mean(geodist), batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_geodist_perc_inf", num_inf / self.b, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None

    # ------------------------------------------------------------------
    # Multi-source steps (permutation-invariant, num_sources == 2)
    # ------------------------------------------------------------------
    def _unpack_multi(self, data):
        """Unpack batch and run forward; returns (class_logits, alpha_true, alpha_pred_or_None)."""
        leadfield, clusters, alpha_true = data[0], data[1], data[2]
        out = self.forward(leadfield)
        if self.predict_alpha:
            class_logits, alpha_pred = out
        else:
            class_logits, alpha_pred = out, None
        return leadfield, clusters, alpha_true, class_logits, alpha_pred

    def _multi_preds(self, class_logits, clusters, swap_mask):
        """Return permutation-matched (pred0, pred1) aligned to (labels0, labels1)."""
        pred0_raw = torch.argmax(class_logits[:, 0, :], dim=1)
        pred1_raw = torch.argmax(class_logits[:, 1, :], dim=1)
        pred0 = torch.where(swap_mask, pred1_raw, pred0_raw)
        pred1 = torch.where(swap_mask, pred0_raw, pred1_raw)
        return pred0, pred1

    def _training_step_multi(self, data, batch_idx):
        _, clusters, alpha_true, class_logits, alpha_pred = self._unpack_multi(data)
        class_loss, swap_mask = self._pi_loss(class_logits, clusters)

        if self.predict_alpha:
            alpha_mae = self.alpha_loss_fn(alpha_pred, alpha_true)
            loss = class_loss + self.alpha_weight * alpha_mae
            self.log("train_alpha_mae", alpha_mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        else:
            loss = class_loss

        pred0, pred1 = self._multi_preds(class_logits, clusters, swap_mask)
        labels0, labels1 = clusters[:, 0], clusters[:, 1]
        strong_acc, weak_acc, weighted_acc = self._strength_metrics(pred0, pred1, labels0, labels1, alpha_true)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        self.log("train_strong_acc", strong_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        self.log("train_weak_acc", weak_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        self.log("train_weighted_acc", weighted_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        return loss

    def _validation_step_multi(self, val_data, batch_idx):
        _, clusters, alpha_true, class_logits, alpha_pred = self._unpack_multi(val_data)
        class_loss, swap_mask = self._pi_loss(class_logits, clusters)

        if self.predict_alpha:
            alpha_mae = self.alpha_loss_fn(alpha_pred, alpha_true)
            val_loss = class_loss + self.alpha_weight * alpha_mae
            self.log("val_alpha_mae", alpha_mae, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        else:
            val_loss = class_loss

        pred0, pred1 = self._multi_preds(class_logits, clusters, swap_mask)
        labels0, labels1 = clusters[:, 0], clusters[:, 1]

        correct_nat = (pred0 == labels0) & (pred1 == labels1)
        correct_swap = (torch.argmax(class_logits[:, 0, :], dim=1) == labels1) & \
                       (torch.argmax(class_logits[:, 1, :], dim=1) == labels0)
        accuracy = torch.where(swap_mask, correct_swap, correct_nat).float().mean()

        strong_acc, weak_acc, weighted_acc = self._strength_metrics(pred0, pred1, labels0, labels1, alpha_true)

        self.log("val_loss", val_loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_strong_acc", strong_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_weak_acc", weak_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_weighted_acc", weighted_acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        if self.cluster_locs is not None:
            d_nat = (torch.norm(self.cluster_locs[pred0] - self.cluster_locs[labels0], dim=1) +
                     torch.norm(self.cluster_locs[pred1] - self.cluster_locs[labels1], dim=1)) / 2
            pred0_raw = torch.argmax(class_logits[:, 0, :], dim=1)
            pred1_raw = torch.argmax(class_logits[:, 1, :], dim=1)
            d_swap = (torch.norm(self.cluster_locs[pred0_raw] - self.cluster_locs[labels1], dim=1) +
                      torch.norm(self.cluster_locs[pred1_raw] - self.cluster_locs[labels0], dim=1)) / 2
            euclid_dist = torch.where(swap_mask, d_swap, d_nat)
            self.log("val_euclid_dist", euclid_dist.mean(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None

    def _test_step_multi(self, test_data, batch_idx):
        _, clusters, alpha_true, class_logits, alpha_pred = self._unpack_multi(test_data)
        class_loss, swap_mask = self._pi_loss(class_logits, clusters)

        if self.predict_alpha:
            alpha_mae = self.alpha_loss_fn(alpha_pred, alpha_true)
            test_loss = class_loss + self.alpha_weight * alpha_mae
            self.log("test_alpha_mae", alpha_mae, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.b, sync_dist=True)
        else:
            test_loss = class_loss

        pred0, pred1 = self._multi_preds(class_logits, clusters, swap_mask)
        labels0, labels1 = clusters[:, 0], clusters[:, 1]

        correct_nat = (pred0 == labels0) & (pred1 == labels1)
        correct_swap = (torch.argmax(class_logits[:, 0, :], dim=1) == labels1) & \
                       (torch.argmax(class_logits[:, 1, :], dim=1) == labels0)
        accuracy = torch.where(swap_mask, correct_swap, correct_nat).float().mean()

        strong_acc, weak_acc, weighted_acc = self._strength_metrics(pred0, pred1, labels0, labels1, alpha_true)

        self.log("test_loss", test_loss, batch_size=self.b, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("hp_metric", accuracy, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_strong_acc", strong_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_weak_acc", weak_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_weighted_acc", weighted_acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        if self.cluster_locs is not None:
            d_nat = (torch.norm(self.cluster_locs[pred0] - self.cluster_locs[labels0], dim=1) +
                     torch.norm(self.cluster_locs[pred1] - self.cluster_locs[labels1], dim=1)) / 2
            pred0_raw = torch.argmax(class_logits[:, 0, :], dim=1)
            pred1_raw = torch.argmax(class_logits[:, 1, :], dim=1)
            d_swap = (torch.norm(self.cluster_locs[pred0_raw] - self.cluster_locs[labels1], dim=1) +
                      torch.norm(self.cluster_locs[pred1_raw] - self.cluster_locs[labels0], dim=1)) / 2
            euclid_dist = torch.where(swap_mask, d_swap, d_nat)
            self.log("test_euclid_dist", euclid_dist.mean(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None


def train_network(clusters, noise_level=0.50, network_name="Linear", batch_size=512, max_epochs=20,
                  learning_rate=1e-4, scheduler="onecycle", pe="none", workers=8, progress_bar=True,
                  num_sources=1, min_dist=20.0, alpha_min=0.25, alpha_max=0.75,
                  predict_alpha=False, alpha_weight=0.1, dropout_num=0):

    leadfield, noise_dir = load_data(clusters)

    cluster_locs, _ = load_locs(clusters)

    dist_table = load_dist_table(clusters)

    electrode_locs = load_electrode_locs()

    num_dipoles = leadfield.shape[1]

    train_size = 100
    val_size = 50
    test_size = 100

    val_noise, test_noise = getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True)

    if num_sources > 1:
        train_dataset = MultiClusterDataset(num_dipoles, leadfield, train_size, num_sources=num_sources,
                                            dist_table=dist_table, min_dist=min_dist,
                                            noise_level=noise_level,
                                            alpha_min=alpha_min, alpha_max=alpha_max)
        validation_dataset = MultiClusterDataset(num_dipoles, leadfield, val_size, num_sources=num_sources,
                                                  dist_table=dist_table, min_dist=min_dist,
                                                  noise_vectors=val_noise,
                                                  alpha_min=alpha_min, alpha_max=alpha_max)
        test_dataset = MultiClusterDataset(num_dipoles, leadfield, test_size, num_sources=num_sources,
                                            dist_table=dist_table, min_dist=min_dist,
                                            noise_vectors=test_noise,
                                            alpha_min=alpha_min, alpha_max=alpha_max)
    else:
        train_dataset = ClusterDataset(num_dipoles, leadfield, train_size, None, noise_level)
        validation_dataset = ClusterDataset(num_dipoles, leadfield, val_size, val_noise, noise_level)
        test_dataset = ClusterDataset(num_dipoles, leadfield, test_size, test_noise, noise_level)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    s = "{s:.2f}".format(s=noise_level).split('.')[1]
    source_tag = f"_{num_sources}src" if num_sources > 1 else ""

    hparams = {"clusters": clusters, "noise_level": noise_level, "network_name": network_name,
                "batch_size": batch_size, "max_epochs": max_epochs, "learning_rate": learning_rate,
                "scheduler": scheduler, "num_sources": num_sources,
                "alpha_min": alpha_min, "alpha_max": alpha_max, "predict_alpha": predict_alpha,
                "dropout_num": dropout_num}

    model = LitNetwork(leadfield.shape[0], leadfield.shape[1], network_name, batch_size, learning_rate,
                        max_epochs, cluster_locs, dist_table, electrode_locs, scheduler, pe, num_sources,
                        predict_alpha=predict_alpha, alpha_weight=alpha_weight, dropout_num=dropout_num)

    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
    log_dir = "my_logs/cluster{c}{s}/noise_{n}".format(c=clusters, s=source_tag, n=s)
    logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=network_name + "_" + scheduler + "_" + str(max_epochs) + "_{lr:.0e}".format(lr=learning_rate)
    )
    logger.log_hyperparams(hparams)

    device = "gpu"  # Use 'mps' for Mac M1/M2, 'gpu' for Windows+Nvidia, 'cpu' otherwise

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=[checkpoint],
                          logger=logger, enable_progress_bar=progress_bar)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model, train_loader, val_loader)

    # ONNX export (single-source only; multi-source output shape varies by network)
    if num_sources == 1:
        onnx_dir = "onnx_models/"
        onnx_path = onnx_dir + network_name + "_noise_{n}.onnx".format(n=s)
        dummy_input = torch.randn(1, leadfield.shape[0], dtype=torch.float32)
        torch.onnx.export(model, dummy_input, onnx_path, input_names=['leadfield'])

    trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='EEG Cluster Prediction')
    parser.add_argument('clusters', type=str, help='Cluster resolution (e.g., "10mm", "5mm")')
    parser.add_argument('noise_level', type=float, help='Noise level (e.g., 0.50)')
    parser.add_argument('--network_name', type=str, default='TransformerEncoder',
                        help='Network: TransformerEncoder (default), Linear')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default="onecycle")
    parser.add_argument('--pe', type=str, default='none', help='Positional encoding: none, enc, dec, both')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no_progress_bar', action="store_false", dest='progress_bar')
    parser.add_argument('--num_sources', type=int, default=1,
                        help='Number of simultaneous sources (1=single-source, 2=dual-source)')
    parser.add_argument('--min_dist', type=float, default=20.0,
                        help='Minimum geodesic distance (mm) between sources for multi-source mode')
    parser.add_argument('--alpha_range', type=float, nargs=2, default=[0.25, 0.75], metavar=('MIN', 'MAX'),
                        help='Alpha (strength ratio) sampling range for multi-source mode (default: 0.25 0.75)')
    parser.add_argument('--predict_alpha', action='store_true',
                        help='Add a head to predict the alpha mixing ratio (multi-source only)')
    parser.add_argument('--alpha_weight', type=float, default=0.1,
                        help='Weight of the alpha prediction loss relative to classification loss (default: 0.1)')
    parser.add_argument('--dropout_num', type=int, default=0,
                        help='Number of electrodes to randomly drop per sample during training (default: 0 = disabled)')
    args = parser.parse_args()

    train_network(
        clusters=args.clusters,
        noise_level=args.noise_level,
        network_name=args.network_name,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        scheduler=args.scheduler,
        pe=args.pe,
        workers=args.workers,
        progress_bar=args.progress_bar,
        num_sources=args.num_sources,
        min_dist=args.min_dist,
        alpha_min=args.alpha_range[0],
        alpha_max=args.alpha_range[1],
        predict_alpha=args.predict_alpha,
        alpha_weight=args.alpha_weight,
        dropout_num=args.dropout_num,
    )
