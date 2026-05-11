
import os
import numpy as np
#from scipy.io import loadmat, savemat
import hdf5storage

import os

def loadmat(filename):
    return hdf5storage.loadmat(filename)

def load_data(clusters, data_path='data/Ernie/'):
    if clusters == "5mm":
        eeg_data = loadmat(data_path + "leadfield_patch5mmdistance.mat")
        leadfield = eeg_data["patch_leadfield"]
        noise_dir = data_path + "noise_vectors_5mm/"
    elif clusters == "10mm":
        eeg_data = loadmat(data_path + "leadfield_patch10mmdistance.mat")
        leadfield = eeg_data["patch_leadfield"]
        noise_dir = data_path + "noise_vectors_10mm/"
    else:
        raise ValueError("Invalid cluster name. Supported: '5mm', '10mm'")

    return leadfield, noise_dir

def load_dist_table(clusters, data_path='data/Ernie/'):
    if clusters == "5mm":
        return loadmat(data_path + "geodesic_distance_matrix_5mm_patches.mat")["distance_matrix"]
    elif clusters == "10mm":
        return loadmat(data_path + "geodesic_distance_matrix_10mm_patches.mat")["distance_matrix"]
    else:
        raise ValueError("Invalid cluster name. Supported: '5mm', '10mm'")

def load_locs(clusters, data_path='data/Ernie/'):
    if clusters == "5mm":
        eeg_data = loadmat(data_path + "leadfield_patch5mmdistance.mat")
    elif clusters == "10mm":
        eeg_data = loadmat(data_path + "leadfield_patch10mmdistance.mat")
    else:
        raise ValueError("Invalid cluster name. Supported: '5mm', '10mm'")

    return eeg_data["patch_center_position"], eeg_data["patch_center_position_normals"]

def load_electrode_locs(data_path='data/Ernie/'):
    eeg_data = loadmat(data_path + "ernie_eeg_simulations.mat")
    electrode_locs = eeg_data["electrodes"]
    return electrode_locs


def getNoiseVectors(leadfield, noise_level, noise_dir, val_size, test_size, use_saved_noise=True):

    s = "{s:.2f}".format(s=noise_level).split('.')[1]

    if use_saved_noise:
        #s = "{s:.2f}".format(s=noise_level).split('.')[1]
        val_fn = noise_dir+"noise_vectors_val_{n}.npy".format(n=s)
        test_fn = noise_dir+"noise_vectors_test_{n}.npy".format(n=s)
        if os.path.exists(val_fn):
            val_noise = np.load(val_fn)
            test_noise = np.load(test_fn)
        else:
            val_noise = makeNoiseVectors(leadfield, val_size, noise_level)
            test_noise = makeNoiseVectors(leadfield, test_size, noise_level)
            np.save(val_fn,val_noise)
            np.save(test_fn,test_noise)
    else:
        val_noise = makeNoiseVectors(leadfield, val_size, noise_level)
        test_noise = makeNoiseVectors(leadfield, test_size, noise_level)

    return val_noise, test_noise


def makeNoiseVectors(leadfield, num_examples, noise_level):

    noise_max = np.amax(np.abs(leadfield),axis=0)
    std_devs = noise_max/2 * noise_level # Make percentage of noise_max 2 Standard Deviations (95%)

    # Initialize an empty list to store all the noise arrays
    noise_list = []

    # Generate num_examples versions of the noise
    for _ in range(num_examples):
        noise = np.random.normal(0, std_devs, leadfield.shape)
        noise_list.append(noise)

    # Concatenate all the noise arrays along the first axis (i.e., vertically stack them)
    noise_vectors = np.stack(noise_list, axis=0)

    return noise_vectors.astype(np.float32)

def get_saved_noise_vectors(noise_dir, noise_level, leadfield, val_size, test_size):
    use_saved_noise = True
    if use_saved_noise:
        s = f"{noise_level:.2f}".split('.')[1]
        val_fn = noise_dir + f"noise_vectors_val_{s}.npy"
        test_fn = noise_dir + f"noise_vectors_test_{s}.npy"
        if os.path.exists(val_fn) and os.path.exists(test_fn):
            val_noise = np.load(val_fn)
            test_noise = np.load(test_fn)
        else:
            val_noise = makeNoiseVectors(leadfield, val_size, noise_level)
            test_noise = makeNoiseVectors(leadfield, test_size, noise_level)
            np.save(val_fn, val_noise)
            np.save(test_fn, test_noise)
    else:
        val_noise = makeNoiseVectors(leadfield, val_size, noise_level)
        test_noise = makeNoiseVectors(leadfield, test_size, noise_level)
    return val_noise, test_noise
