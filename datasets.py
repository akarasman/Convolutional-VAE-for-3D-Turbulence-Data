import numpy as np
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import pdb
import torch

from PIL import Image
from collections import defaultdict
from skimage.transform import resize
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Callable, List, Optional, Tuple

class Field3DDataset(Dataset):
    """
    Field3DDataset is a custom PyTorch Dataset class for handling 3D CFD (Computational Fluid Dynamics) data.
    Attributes:
        data_directory (str): Path to the directory containing subfolders with the npy files.
        n_voxels (int): Number of voxels for resizing the cubes.
        transforms (callable, optional): Optional transform to be applied on a sample.
        data_len (int): Length of the dataset.
        mean (torch.Tensor): Mean of the dataset along the C channels.
        std (torch.Tensor): Standard deviation of the dataset along the C channels.
    Methods:
        __compute_mean_std_dataset(data):
            Computes the mean and standard deviation of the dataset along the C channels.
        __standardize_cubes(data):
            Standardizes the data to have mean 0 and std 1.
        __minmax_normalization(data):
            Performs MinMax normalization on the given array.
        __scale_by(cube, factor):
            Scales the array by a given factor.
        __getitem__(index):
            Returns a tensor cube of shape (C, n_voxels, n_voxels, n_voxels) normalized by subtracting mean and dividing std of dataset computed beforehand.
        __len__():
            Returns the length of the dataset.
    """

    def __init__(self, data_directory: str, n_voxels: int = 21, channels: int = 3, transforms: Optional[Callable] = None) -> None:
        """
        data_directory: path to directory that contains subfolders with the npy files
        Subfolders are folders containing each component of velocity: extract_cubes_U0_reduced
        """

        print("[INFO] started instantiating 3D CFD pytorch dataset")

        self.data_directory = data_directory
        self.n_voxels = n_voxels
        self.channels = channels
        self.transforms = transforms

        # List all data files
        self.data_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pt')]
        self.data_len = len(self.data_files)

        # Compute mean and std on-the-fly if needed
        self.mean, self.std = self.__compute_mean_std_dataset()

    def __compute_mean_std_dataset(self):
        """
        Computes the mean and standard deviation for each channel of the dataset.
        This method calculates the mean and standard deviation across the entire dataset.
        """
        mean = torch.zeros(self.n_voxels, self.n_voxels, self.n_voxels, self.channels)
        std = torch.zeros(self.n_voxels, self.n_voxels, self.n_voxels, self.channels)

        for data_file in self.data_files:
            data_cube = torch.load(data_file)
            mean += data_cube
        for data_file in self.data_files:
            data_cube = torch.load(data_file)
            std += (data_cube - mean) ** 2

        mean /= self.data_len
        std /= self.data_len
        
        return mean, std

    def __standardize_cubes(self, data):
        """
        Standardizes the input data cubes by subtracting the mean and dividing by the standard deviation.
        """
        return (data - self.mean) / self.std

    def __minmax_normalization(self, data):
        """
        Performs MinMax normalization on given array. Range [0, 1]
        """
        data_min = torch.min(data)
        data_max = torch.max(data)
        return (data - data_min) / (data_max - data_min)

    def __scale_by(self, cube, factor):
        mean = torch.mean(cube)
        return (cube - mean) * factor + mean

    def __getitem__(self, index):
        """
        Returns a tensor cube of shape (C, n_voxels, n_voxels, n_voxels) normalized by subtracting mean and dividing std of dataset computed beforehand.
        """
        data_file = self.data_files[index]
        data_cube = torch.load(data_file)

        # Standardize the data
        cube_tensor = self.__standardize_cubes(data_cube)
        cube_minmax = self.__minmax_normalization(cube_tensor)

        # Very strange transformation, not sure what it does
        cube_clamped = torch.clamp(cube_minmax - 0.1, 0, 1)
        cube_transformed = torch.clamp(self.__scale_by(cube_clamped**0.4, 2)-0.1, 0, 1)
        cube_resized = torch.tensor(resize(cube_transformed.numpy(), [ self.n_voxels ] * 3, mode='constant'))

        # swap axes from torch shape (21, 21, 21, 3) to torch shape (3, 21, 21, 21) this is for input to Conv3D
        cube_reshaped = cube_resized.permute(3, 0, 1, 2)

        if self.transforms:
            cube_reshaped = self.transforms(cube_reshaped)

        return cube_reshaped

    def __len__(self):
        return self.data_len