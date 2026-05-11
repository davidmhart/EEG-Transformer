
# Code taken from https://github.com/tatp22/multidim-positional-encoding

import torch
import torch.nn as nn
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        #if len(tensor.shape) != 3:
        #    raise RuntimeError("The input tensor has to be 3d (batch_size, num_elements, channels)!")

        #batch_size, num_elements, orig_ch = tensor.shape
        num_elements, _ = positions.shape

        # Extract x, y, z positions from the input positions
        pos_x = positions[:, 0]  # x coordinates
        pos_y = positions[:, 1]  # y coordinates
        pos_z = positions[:, 2]  # z coordinates

        # Apply sinusoidal encoding based on positions
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)  # Shape: (num_elements, channels)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)

        # Get positional embeddings for each dimension
        emb_x = get_emb(sin_inp_x).unsqueeze(1)  # Shape: (num_elements, 1, channels)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z).unsqueeze(1)

        # Combine the embeddings for x, y, z dimensions
        emb = torch.zeros((num_elements, self.channels * 3), device=positions.device, dtype=positions.dtype)
        emb[:, :self.channels] = emb_x.squeeze(1)
        emb[:, self.channels:2 * self.channels] = emb_y.squeeze(1)
        emb[:, 2 * self.channels:] = emb_z.squeeze(1)

        # Prune the embeddings to the original number of channels
        emb = emb[:, :self.org_channels]

        return emb
