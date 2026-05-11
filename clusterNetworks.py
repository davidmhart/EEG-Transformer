
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_encoding_3d import PositionalEncoding3D


def _drop_input_electrodes(x, num_keep):
    """Zero out a random subset of electrodes in the input vector.
    Zeroing an input electrode zeros all paths from that node through the network,
    since every downstream computation is multiplied by the zeroed value.
    x: (batch, num_electrodes)
    Returns x with (num_electrodes - num_keep) electrodes set to zero per sample.
    """
    B, E = x.shape
    mask = torch.zeros(B, E, device=x.device)
    ids_keep = torch.argsort(torch.rand(B, E, device=x.device), dim=1)[:, :num_keep]
    mask.scatter_(1, ids_keep, 1.0)
    return x * mask


def _drop_electrode_tokens(x_tokens, num_keep):
    """Remove a random subset of electrode token embeddings from the sequence.
    Unlike zeroing, this removes the tokens entirely so the transformer never attends to them.
    x_tokens: (batch, num_electrodes, d_model)
    Returns (batch, num_keep, d_model).
    """
    B, E, D = x_tokens.shape
    ids_keep = torch.argsort(torch.rand(B, E, device=x_tokens.device), dim=1)[:, :num_keep]
    return torch.gather(x_tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))


class LinearNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_num=0):
        super(LinearNetwork, self).__init__()
        self.dropout_num = int(dropout_num)
        self.num_keep = in_channels - self.dropout_num
        if self.num_keep <= 0:
            raise ValueError(f"dropout_num={dropout_num} must be less than in_channels={in_channels}")

        self.linear1 = nn.Linear(in_channels, 100)
        self.linear2 = nn.Linear(100, 1000)
        self.linear3 = nn.Linear(1000, out_channels)

    def forward(self, x):
        if self.dropout_num > 0:
            x = _drop_input_electrodes(x, self.num_keep)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for single- or multi-source EEG cluster prediction.

    Uses one learned token per source. With num_sources=1 the output shape is
    (batch, out_channels); with num_sources>1 it is (batch, num_sources, out_channels).
    dropout_num electrodes are randomly removed from the input sequence each forward pass.
    """
    def __init__(self, in_channels, out_channels, electrode_locs, num_sources=1,
                 d_model=512, nhead=8, num_encoder_layers=3,
                 with_pos_enc=False, dropout=0.1, dropout_num=0):
        super(TransformerEncoder, self).__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.num_sources = num_sources
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
        # One learned token per source; the transformer reads off the first num_sources positions
        self.learned_tokens = nn.Parameter(torch.rand(num_sources, d_model))
        self.output_projection = nn.Linear(d_model, out_channels)

    def forward(self, x):
        if self.with_pos_enc:
            x = x.unsqueeze(2)
            x = self.input_projection(x)
            encodings = self.pe(self.electrode_locs).unsqueeze(0).expand(x.size(0), -1, -1)
            x_tokens = x + encodings
        else:
            x_tokens = self.input_projection(
                torch.cat((x.unsqueeze(2), self.electrode_locs.expand(x.shape[0], -1, -1)), dim=2)
            )

        if self.dropout_num > 0:
            x_tokens = _drop_electrode_tokens(x_tokens, self.num_keep)

        tokens = self.learned_tokens.unsqueeze(0).expand(x_tokens.size(0), -1, -1)
        x_out = self.transformer_encoder(torch.cat((tokens, x_tokens), dim=1))

        # Extract the num_sources output positions and project to class scores
        out = self.output_projection(x_out[:, :self.num_sources, :])
        # Squeeze source dimension for single-source to keep downstream shapes unchanged
        return out.squeeze(1) if self.num_sources == 1 else out
