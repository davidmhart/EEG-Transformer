
from torch.utils.data import Dataset
import torch
import numpy as np

class ClusterDataset(Dataset):
    def __init__(self, num_dipoles, leadfield, num_examples, noise_vectors=None, noise_level=0.0):
        super(ClusterDataset, self).__init__()

        self.num_dipoles = num_dipoles
        self.num_examples = num_examples

        self.length = num_dipoles*num_examples

        self.leadfield = leadfield
        
        if noise_vectors is None:
            self.noise_vectors = None
            noise_max = np.amax(np.abs(leadfield),axis=0)
            self.std_devs = noise_max/2 * noise_level
        else:
            self.noise_vectors = torch.tensor(noise_vectors, dtype=torch.float32)
    

    def __getitem__(self, idx):
        
        cluster = idx % self.num_dipoles

        leadfield_dipole = self.leadfield[:,cluster]

        # Convert data types to single precision floats and longs
        leadfield_dipole = torch.tensor(leadfield_dipole, dtype=torch.float32)

        # Add noise to the leadfield
        if self.noise_vectors is None:
            noise = torch.normal(0,self.std_devs[cluster],size=leadfield_dipole.shape)
        else:
            noise = self.noise_vectors[idx // self.num_dipoles,:,cluster]

        leadfield_dipole = leadfield_dipole + noise

        return leadfield_dipole, torch.tensor(cluster, dtype=torch.long)
    

    def __len__(self):
        return self.length
    
class MultiClusterDataset(Dataset):
    """
    Dataset for multi-source cluster prediction with alpha (strength) mixing.

    Each sample mixes the leadfields of two randomly chosen (sufficiently separated)
    clusters with a random strength ratio alpha ~ Uniform(alpha_min, alpha_max):
        signal = alpha * lf0 + (1 - alpha) * lf1 + noise
    alpha=0.5 is equal strength; values away from 0.5 create a strong/weak pair.

    dist_table (num_dipoles x num_dipoles distances in mm) enforces a minimum separation;
    pass None to skip the distance constraint.

    noise_mode:
      - "sensor": one noise draw added to the summed signal
      - "per_source": independent noise draws per source, then summed

    Returns: (signal, clusters, alpha) where alpha is a scalar float32 tensor.
    """
    def __init__(self, num_dipoles, leadfield, num_examples, num_sources=2, dist_table=None,
                 min_dist=20.0, noise_vectors=None, noise_level=0.0, noise_mode="sensor",
                 alpha_min=0.25, alpha_max=0.75):
        super(MultiClusterDataset, self).__init__()
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

        if noise_vectors is None:
            self.noise_vectors = None
            noise_max = np.amax(np.abs(leadfield), axis=0)
            self.std_devs = noise_max / 2 * noise_level
        else:
            self.noise_vectors = torch.tensor(noise_vectors, dtype=torch.float32)

    def __getitem__(self, idx):
        example_idx = idx // self.num_dipoles
        for _ in range(1000):
            clusters = np.random.choice(self.num_dipoles, self.num_sources, replace=False)
            if self.dist_table is None or self.dist_table[clusters[0], clusters[1]] >= self.min_dist:
                break

        c0, c1 = int(clusters[0]), int(clusters[1])
        alpha = float(np.random.uniform(self.alpha_min, self.alpha_max))

        lf0 = torch.tensor(self.leadfield[:, c0], dtype=torch.float32)
        lf1 = torch.tensor(self.leadfield[:, c1], dtype=torch.float32)
        # Normalize so the dominant source is always at full amplitude.
        # At alpha=0.5 this reduces to lf0 + lf1, matching the equal-strength baseline.
        scale = 1.0 / max(alpha, 1.0 - alpha)
        lf = scale * (alpha * lf0 + (1.0 - alpha) * lf1)

        if self.noise_vectors is None:
            if self.noise_mode == "per_source":
                noise = (torch.normal(0.0, float(self.std_devs[c0]), size=lf.shape) +
                         torch.normal(0.0, float(self.std_devs[c1]), size=lf.shape))
            else:
                noise = torch.normal(0.0, float(self.std_devs[c0]), size=lf.shape)
        else:
            ni = example_idx % self.noise_vectors.shape[0]
            if self.noise_vectors.ndim == 3:
                if self.noise_mode == "per_source":
                    noise = self.noise_vectors[ni, :, c0] + self.noise_vectors[ni, :, c1]
                else:
                    noise = self.noise_vectors[ni, :, c0]
            else:
                noise = self.noise_vectors[ni, :]

        return (lf + noise,
                torch.tensor(clusters, dtype=torch.long),
                torch.tensor(alpha, dtype=torch.float32))

    def __len__(self):
        return self.length


