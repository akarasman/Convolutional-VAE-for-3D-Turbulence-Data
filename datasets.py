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


class CFD3DDataset(Dataset):
    
    """
    CFD3DDataset is a custom PyTorch Dataset class for handling 3D CFD (Computational Fluid Dynamics) data.
    Attributes:
        data_directory (str): Path to the directory containing subfolders with the npy files.
        no_simulations (int): Number of simulations.
        simulation_timesteps (int): Number of timesteps per simulation.
        transforms (callable, optional): Optional transform to be applied on a sample.
        cubes_U_all_channels (list): List of numpy arrays, each representing a cube with shape (21, 21, 21, 3).
        data_len (int): Length of the dataset.
        stacked_cubes (numpy.ndarray): Numpy array of stacked cubes with shape (9600, 21, 21, 21, 3).
        mean (torch.Tensor): Mean of the dataset along the 3 channels.
        std (torch.Tensor): Standard deviation of the dataset along the 3 channels.
        standardized_cubes (numpy.ndarray): Standardized cubes with mean 0 and std 1.
    Methods:
        __load_3D_cubes(self, data_directory):
            Loads 3D CFD data from the specified directory into a dictionary.
        __compare_U_sim_keys(self, cube1, cube2):
            Compares keys of two dictionaries to ensure they have the same simulation parameters.
        __merge_velocity_components_into_dict(self, cubes_U0, cubes_U1, cubes_U2):
            Merges velocity components U0, U1, U2 into a single dictionary based on simulation keys.
        __concatenate_3_velocity_components(self, cubes_dict):
            Concatenates the 3 velocity components into a list of numpy arrays.
        __compute_mean_std_dataset(self, data):
            Computes the mean and standard deviation of the dataset along the 3 channels.
        __standardize_cubes(self, data):
            Standardizes the data to have mean 0 and std 1.
        __minmax_normalization(self, data):
            Performs MinMax normalization on the given array.
        __scale_by(self, arr, fac):
            Scales the array by a given factor.
    """

    def __init__(self, data_directory, no_simulations, simulation_timesteps, transforms=None):
        """
        data_directory: path to directory that contains subfolders with the npy files
        Subfolders are folders containing each component of velocity: extract_cubes_U0_reduced
        """

        print()
        print("[INFO] started instantiating 3D CFD pytorch dataset")

        self.data_directory = data_directory
        self.no_simulations = no_simulations # 96
        self.simulation_timesteps = simulation_timesteps # 100
        self.transforms = transforms

        # data_dir = "../cfd_data/HVAC_DUCT/cubes/coords_3d/"
        data_directory_U0 = self.data_directory + "extract_cubes_U0_reduced/"
        data_directory_U1 = self.data_directory + "extract_cubes_U1_reduced/"
        data_directory_U2 = self.data_directory + "extract_cubes_U2_reduced/"

        # read cubes data from directories
        cubes_U0_dict = self.__load_3D_cubes(data_directory_U0)
        cubes_U1_dict = self.__load_3D_cubes(data_directory_U1)
        cubes_U2_dict = self.__load_3D_cubes(data_directory_U2)

        # compare all folders have same simulation parameters
        if self.__compare_U_sim_keys(cubes_U0_dict, cubes_U1_dict) and \
           self.__compare_U_sim_keys(cubes_U0_dict, cubes_U2_dict) and \
           self.__compare_U_sim_keys(cubes_U1_dict, cubes_U2_dict):
            print("[INFO] all folders have same keys (simulations)")
        else:
            print("[INFO] the folders don't have the same keys (simulations)")
            quit()

        # concatenate all velocity components into one dictionary data structure
        cubes_U_all_dict = self.__merge_velocity_components_into_dict(cubes_U0_dict, cubes_U1_dict, cubes_U2_dict)

        # creates a list of length timesteps x simulations, each element is a numpy array with cubes size (21,21,21,3)
        # cubes_U_all_channels: 9600 with shape (21,21,21,3)
        self.cubes_U_all_channels = self.__concatenate_3_velocity_components(cubes_U_all_dict)
        print("[INFO] cubes dataset length:", len(self.cubes_U_all_channels))
        print("[INFO] single cube shape:", self.cubes_U_all_channels[0].shape)
        self.data_len = len(self.cubes_U_all_channels)

        # stack all cubes in a final numpy array numpy (9600, 21, 21, 21, 3)
        self.stacked_cubes = torch.stack(self.cubes_U_all_channels, 0)

        print()
        print("[INFO] mean and std of the cubes dataset along 3 channels")
        # note: not using mean and std separately, just calling them in standardize function (below)
        # note: only standardize data to mean 0 and std 1
        self.mean, self.std = self.__compute_mean_std_dataset(self.stacked_cubes)
        print("mean:", self.mean)
        print("std:", self.std)

        # standardize data from here
        print()
        print("[INFO] standardize data to mean 0 and std 1")
        self.standardized_cubes = self.__standardize_cubes(self.stacked_cubes)
        print("mean after standardization:", self.standardized_cubes.mean(axis=(0,1,2,3)))
        print("std after standardization:", self.standardized_cubes.std(axis=(0,1,2,3)))

        print()
        print("[INFO] finished instantiating 3D CFD pytorch dataset")

    def __load_3D_cubes(self, data_directory):
        
        """
        Loads 3D CFD data from .npy files in the specified directory and stores them in a dictionary.
        Args:
            data_directory (str): The directory containing the .npy files.
        Returns:
            dict: A dictionary where keys are the .npy file names (excluding the first two characters) 
                  and values are the loaded numpy arrays of size (21, 21, 21, 100).
        """

        cubes = {}

        for filename in os.listdir(data_directory):
            if filename.endswith(".npy"):
                # set key without Ui character (for later key matching)
                cubes[filename[2:]] = (np.load(data_directory + "/" + filename))

        return cubes

    def __compare_U_sim_keys(self, cube1, cube2):

        """
        Compares the keys of two dictionaries representing velocity components
        to ensure they have the same simulation parameters based on the .npy file names.
        Args:
            cube1 (dict): The first dictionary containing velocity component data.
            cube2 (dict): The second dictionary containing velocity component data.
        Returns:
            bool: True if both dictionaries have the same number of matched keys as the 
                  expected number of simulations (`self.no_simulations`), False otherwise.
        """
        matched_keys = 0
        for key in cube1:
            if key in cube2:
                matched_keys += 1

        if matched_keys == self.no_simulations:
            return True
        else:
            return False

    def __merge_velocity_components_into_dict(self, cubes_U0, cubes_U1, cubes_U2):
        """
        Concatenates all velocity components U0, U1, U2 based on
        Args:
            cubes_U0 (dict): Dictionary containing the first velocity component.
            cubes_U1 (dict): Dictionary containing the second velocity component.
            cubes_U2 (dict): Dictionary containing the third velocity component.
        Returns:
            defaultdict: A dictionary where each key corresponds to a simulation name
                         and the value is a list containing the three velocity components.
        """
        cubes_U = defaultdict(list)

        for d in (cubes_U0, cubes_U1, cubes_U2): # you can list as many input dicts as you want here
            for key, value in d.items():
                cubes_U[key].append(value)

        # this returns a list of sublists, each sublists contains 3 arrays (corresponding to U0, U1, U2)
        print("[INFO] velocity components concatenated into list")
        return cubes_U

    def __concatenate_3_velocity_components(self, cubes_dict):

        """
        Concatenates the three velocity components (U0, U1, U2) from the input dictionary of cubes
        into a single array with three channels for each timestep.
        Args:
            cubes_dict (dict): A dictionary where keys are identifiers and values are 4D numpy arrays
                               representing velocity components. Each array has the shape 
                               (x_dim, y_dim, z_dim, timesteps).
        Returns:
            list: A list of 4D numpy arrays, each with shape (x_dim, y_dim, z_dim, 3), where the last 
                  dimension represents the concatenated velocity components for each timestep.
        """
        cubes_3_channels = []

        for key, value in cubes_dict.items():
            # split temporal dependency of simulations
            for timestep in range(0, self.simulation_timesteps):
                # fetch velocity compponents
                U0 = cubes_dict[key][0][:,:,:,timestep] # one cube, three channels, one time step
                U1 = cubes_dict[key][1][:,:,:,timestep]
                U2 = cubes_dict[key][2][:,:,:,timestep]

                # concatenate as channels (21, 21, 21, 3)
                U_all_channels = np.concatenate((U0[...,np.newaxis],
                                                 U1[...,np.newaxis],
                                                 U2[...,np.newaxis]),
                                                 axis=3)

                cubes_3_channels.append(U_all_channels)

        return cubes_3_channels

    def __compute_mean_std_dataset(self, data):
        

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

        mean = 0.
        std = 0.

        mean = torch.mean(data, dim=(0,1,2,3), dtype=torch.float64) 
        std = torch.std(data, dim=(0,1,2,3), dtype=torch.float64)

        return mean, std

    def __standardize_cubes(self, data):
        
        """
        Standardizes the input data cubes by subtracting the mean and dividing by the standard deviation 
        along the specified axes.
        Parameters:
        data (numpy.ndarray): The input data array with shape (9600, 21, 21, 21, 3).
        Returns:
        numpy.ndarray: The standardized data array with the same shape as the input.
        """

        # (9600, 21, 21, 21, 3)
        # means = [7.5, 6.3, 1.2]
        return (data - data.mean(dim=(0,1,2,3), keepdim=True)) / data.std(dim=(0,1,2,3), keepdim=True)

    def ___getitem__(self, index):

        """
        Returns a tensor cube of shape (3,21,21,21) normalized by
        substracting mean and dividing std of dataset computed beforehand.
        """

        single_cube_tensor = self.standardized_cubes[index] # (21, 21, 21, 3)

        # min-max normalization, clipping and resizing
        single_cube_minmax = self.__minmax_normalization(single_cube_tensor) # (custom function)
        single_cube_transformed = torch.clamp(self.__scale_by(torch.clamp(single_cube_minmax-0.1, 0, 1)**0.4, 2)-0.1, 0, 1) # (from tutorial)
        single_cube_resized = torch.from_numpy(resize(single_cube_transformed.numpy(), (21, 21, 21), mode='constant')) # (21,21,21)

        # swap axes from torch shape (21, 21, 21, 3) to torch shape (3, 21, 21, 21) this is for input to Conv3D
        single_cube_reshaped = single_cube_resized.permute(3, 0, 1, 2)

        # NOTE: not applying ToTensor() because it only works with 2D images
        # if self.transforms is not None:
            # single_cube_tensor = self.transforms(single_cube_normalized)
            # single_cube_tensor = self.transforms(single_cube_PIL)

        return single_cube_tensor

    def __minmax_normalization(self, data):
       
       """
       Performs MinMax normalization on given array. Range [0, 1]
       """

       # data shape (21, 21, 21, 3)
       data_min = np.min(data, axis=(0,1,2))
       data_max = np.max(data, axis=(0,1,2))

       return (data-data_min)/(data_max - data_min)
       data_min = torch.min(data, dim=(0,1,2)).values
       data_max = torch.max(data, dim=(0,1,2)).values

       return (data - data_min) / (data_max - data_min)

    def ___len__(self):
        return self.data_len
