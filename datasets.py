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
        stacked_cubes (torch.Tensor): Tensor of stacked cubes with shape (N, n_voxels, n_voxels, n_voxels, n_voxels).
        mean (torch.Tensor): Mean of the dataset along the C channels.
        std (torch.Tensor): Standard deviation of the dataset along the C channels.
        standardized_cubes (torch.Tensor): Standardized cubes with mean 0 and std 1.
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

    def __init__(self, data_directory: str, n_voxels=21, transforms: Optional[Callable] = None) -> None:
        """
        data_directory: path to directory that contains subfolders with the npy files
        Subfolders are folders containing each component of velocity: extract_cubes_U0_reduced
        """

        print("[INFO] started instantiating 3D CFD pytorch dataset")

        self.data_directory = data_directory
        self.n_voxels = n_voxels
        self.transforms = transforms

        # load 3D CFD data from .npy files in the specified directory
        data_cubes: List[torch.Tensor] = []
        print(data_directory)
        for i, folder in enumerate(os.listdir(data_directory)):
            data_file = os.path.join(data_directory, f'sample_{i}.pt')
            data_cube = torch.load(data_file)
            data_cube.unsqueeze(0)
            data_cubes.append(data_cube)

        self.data_len = len(data_cubes)

        # stack all cubes in a final numpy array numpy (9600, 21, 21, 21, 3)
        self.stacked_cubes = torch.stack(data_cubes, 0)

        # note: not using mean and std separately, just calling them in standardize function (below)
        # note: only standardize data to mean 0 and std 1
        self.mean, self.std = self.__compute_mean_std_dataset(self.stacked_cubes)

        # standardize data from here
        self.standardized_cubes = self.__standardize_cubes(self.stacked_cubes)

    @staticmethod
    def __compute_mean_std_dataset(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Computes the mean and standard deviation for each channel of a 3D cube dataset.
        This method calculates the mean and standard deviation across the entire dataset,
        rather than on individual batches, for each of the three channels in the 3D data.
        References:
        - https://stackoverflow.com/questions/47124143/mean-value-of-each-channel-of-several-images
        - https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
        Args:
            data (numpy.ndarray): The 3D cube dataset with shape (N, D, H, W, C), where
                                  N is the number of samples, D is the depth, H is the height,
                                  W is the width, and C is the number of channels.
        Returns:
            tuple: A tuple containing two 1D torch tensors:
                   - mean (torch.Tensor): The mean values for each channel.
                   - std (torch.Tensor): The standard deviation values for each channel.
        """

        mean = torch.mean(data, dim=(0,1,2,3)) 
        std = torch.std(data, dim=(0,1,2,3))

        return mean, std

    @staticmethod
    def __standardize_cubes(data: torch.Tensor) -> torch.Tensor:
        """
        Standardizes the input data cubes by subtracting the mean and dividing by the standard deviation 
        along the specified axes.
        Parameters:
        data (torch.tensor): The input data tensor with shape (X, Y, Z, C).
        Returns:
        torch.tensor: The standardized data tensor with the same shape as the input.
        """


        return (data - data.mean(dim=(0,1,2,3), keepdim=True)) / data.std(dim=(0,1,2,3), keepdim=True)

    def __getitem__(self, index: int) -> torch.Tensor:

        """
        Returns a tensor cube of shape (3,21,21,21) normalized by
        substracting mean and dividing std of dataset computed beforehand.
        """

        cube_tensor = self.standardized_cubes[index]

        # min-max normalization, clipping and resizing
        cube_minmax = self.__minmax_normalization(cube_tensor)

        # Very strange transformation, not sure what it does
        cube_clamped = torch.clamp(cube_minmax - 0.1, 0, 1)
        cube_transformed = torch.clamp(self.__scale_by(cube_clamped**0.4, 2)-0.1, 0, 1)
        cube_resized = torch.tensor(resize(cube_transformed.numpy(), [ self.n_voxels ] * 3, mode='constant'))

        # swap axes from torch shape (21, 21, 21, 3) to torch shape (3, 21, 21, 21) this is for input to Conv3D
        cube_reshaped = cube_resized.permute(3, 0, 1, 2)

        # NOTE: not applying ToTensor() because it only works with 2D images
        # if self.transforms is not None:
        #     cube_tensor = self.transforms(single_cube_normalized)
        #     cube_tensor = self.transforms(single_cube_PIL)

        return cube_reshaped

    @staticmethod
    def __scale_by(cube: torch.Tensor, factor: float) -> torch.Tensor:
        mean = cube.mean()
        return (cube-mean)*factor + mean

    @staticmethod
    def __minmax_normalization(data: torch.Tensor) -> torch.Tensor:
        """
        Performs MinMax normalization on given array. Range [0, 1]
        """

        data_min, _ = torch.min(data, dim=0)
        data_max, _ = torch.max(data, dim=0)

        return (data-data_min) / (data_max - data_min)

    def __len__(self) -> int:
        return self.data_len
