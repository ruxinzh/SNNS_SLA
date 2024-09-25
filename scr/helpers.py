import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import comb
import itertools


def steering_vector(N, deg):
    """
    Calculate the steering vector for a uniform linear array using given antenna configuration.
    
    Args:
        N (int): Number of antenna elements.
        deg (float): Angle of arrival in degrees.

    Returns:
        torch.Tensor: The steering vector as a complex-valued tensor.
    """
    d = 0.5  # Element spacing (in units of wavelength)
    wavelength = 1.0  # Wavelength of the signal (same units as d)
    k = 2 * torch.pi / wavelength  # Wavenumber
    n = torch.arange(0, N).view(N, 1)  # Antenna element indices [0, 1, ..., N-1]
    theta = deg * torch.pi / 180  # Convert degrees to radians
    phases = k * d * n * torch.sin(theta)  # Phase shift for each element

    return torch.exp(1j * phases)  # Complex exponential for each phase shift



def generate_complex_signal(N=10, snr_db=10, deg=torch.tensor([30]),amp=torch.tensor([1])):
    """
    Generates a complex-valued signal for an array of N antenna elements.

    Args:
        N (int): Number of antenna elements.
        snr_db (float): Signal-to-Noise Ratio in decibels.
        deg (tensor): Angle of arrival in degrees.

    Returns:
        torch.Tensor: Complex-valued tensor of shape (N, 1) representing the received signals.
    """
    a_theta = steering_vector(N, deg)
    phase = (amp * torch.exp(2j * torch.pi * torch.randn(a_theta.size()[1]))).view(-1, 1)
    signal = torch.matmul(a_theta.to(phase.dtype), phase)
    # signal_power = torch.mean(torch.abs(signal)**2)
    signal_power = torch.min(amp)**2
    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_real = torch.sqrt(noise_power / 2) * torch.randn_like(signal.real)
    noise_imag = torch.sqrt(noise_power / 2) * torch.randn_like(signal.imag)
    noise = torch.complex(noise_real, noise_imag)

    return signal + noise  


def generate_label(angles, deg_indices):
    """
    Generate one-hot encoded labels for the given degrees.
    
    Args:
        degrees (tensor): Target angles in degrees.

    Returns:
        torch.Tensor: One-hot encoded labels.
    """
    # a_theta = steering_vector(N, degrees)
    # amps = amps.view(-1, 1)
    # labels = torch.matmul(a_theta, amps.to(a_theta.dtype))
    labels = torch.zeros_like(angles)
    labels[deg_indices] =  1 
    return labels

def generate_combinations(output_size, max_targets):
    all_combinations = []
    for num_targets in range(1, max_targets + 1):
        # Generate all combinations for the current number of targets
        for combination in itertools.combinations(range(output_size), num_targets):
            # Create a zero-initialized list of the appropriate length
            outcome = [0] * output_size
            # Set indices in the combination to 1
            for index in combination:
                outcome[index] = 1
            all_combinations.append(outcome)
    return all_combinations


def generate_data(N, num_samples=1, max_targets=3, folder_path='/content/drive/MyDrive/Asilomar2024/data/'):
    """
    Generate dataset with random number of targets and varying SNR levels.
    
    Args:
        N (int): Number of antenna elements.
        num_samples (int): Number of samples to generate for each SNR level.
        max_targets (int): Maximum number of targets.
        folder_path (str): Base folder path for saving data.

    Returns:
        int: Always returns 0. Data saved in specified directory.
    """
    
    # angles = torch.rad2deg(torch.asin(torch.linspace(-1,1,1024)))
    angles = torch.arange(-60, 61, 1)
    # Parameters
    N_output = angles.shape[0]  # Number of possible positions
    # Calculate the sum of combinations
    all_labels_comb = generate_combinations(N_output, max_targets)
    
    signal_folder = os.path.join(folder_path, 'signal')
    label_folder = os.path.join(folder_path, 'label')
    os.makedirs(signal_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
      
    all_signals, all_labels = [], []
    for i in tqdm(range(len(all_labels_comb))):
        noisy_signal = []
        label = torch.tensor(all_labels_comb[i])
        deg_indices = torch.where(label)[0]
        degs = angles[deg_indices]
        num_targets = deg_indices.shape[0] 
        N_samples = num_samples 
        # torch.randint(1,num_samples,(1,))
        for _ in range(N_samples):
            snr_db = 30 * torch.rand(1)
            amp = 0.5 + 0.5 * torch.rand(num_targets-1)
            amps = torch.cat((amp, torch.tensor([1.0])))               
            noisy_signal_1 = generate_complex_signal(N=N, snr_db=snr_db, deg=degs, amp=amps)
            noisy_signal.append(noisy_signal_1)
        all_signals.append(noisy_signal)
        all_labels.append(label)
    torch.save(all_signals, os.path.join(signal_folder, 'signals.pt'))
    torch.save(all_labels, os.path.join(label_folder, 'labels.pt'))
    return None  

class SiameseDataset(Dataset):
    def __init__(self, file_paths, label_paths):
        self.data = torch.load(file_paths[0])
        self.labels = torch.load(label_paths[0])
        self.indices = np.array(len(self.data))

    def __getitem__(self, index):
        # Anchor image
        sig_list, label1 = self.data[index], self.labels[index]
        # Choose a positive or negative pair randomly
        N_samp = len(sig_list)-1
        if random.random() < 0.5:
            # Positive sample
            idx1 = random.randint(0, N_samp)
            idx2 = random.randint(0, N_samp)
            sig1 = self.data[index][idx1]
            sig2 = self.data[index][idx2]
            trt = torch.tensor(1, dtype=torch.float32)
        else:
            # Negative sample
            trt = torch.tensor(0, dtype=torch.float32)
            idx1 = random.randint(0, N_samp)
            sig1 = self.data[index][idx1]
            while True:
                index2 = random.randint(0, self.indices-1)                
                # label2 = self.labels[index2]
                # true_indices = torch.where(label1)[0]               
                if index2 != index:
                    sig_list2 = self.data[index2]
                    N_samp2 = len(sig_list2)-1       
                    idx2 = random.randint(0, N_samp2)
                    sig2 = self.data[index2][idx2]
                    break

        return sig1, sig2, label1, trt

    def __len__(self):
        return len(self.data)


def create_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Create a DataLoader for batching and shuffling the dataset.

    Args:
        data_path (str): Path to the directory containing the data files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    signal_dir_path = os.path.join(data_path, "signal")
    label_dir_path = os.path.join(data_path, "label")
    signal_files = [os.path.join(signal_dir_path, f) for f in os.listdir(signal_dir_path) if 'signals' in f]
    label_files = [os.path.join(label_dir_path, f) for f in os.listdir(label_dir_path) if 'labels' in f]
    dataset = SiameseDataset(sorted(signal_files), sorted(label_files))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 


def omp(D, x, sparsity_level):
    """
    Orthogonal Matching Pursuit (OMP) algorithm.

    Args:
    D (numpy.ndarray): Dictionary matrix (sensing matrix) with dimensions (m, n).
    x (numpy.ndarray): Measurement vector with dimensions (m, 1).
    sparsity_level (int): The expected sparsity level (number of non-zero coefficients).

    Returns:
    numpy.ndarray: Sparse representation vector with dimensions (n, 1).
    """
    # Initialize residual, support, and solution vector
    residual = x.copy()
    support = []
    solution = np.zeros(D.shape[1])

    for _ in range(sparsity_level):
        # Project the residual on the dictionary and find the index of max correlation
        correlations = D.T @ residual
        best_index = np.argmax(np.abs(correlations))
        
        # Update the support set
        if best_index not in support:
            support.append(best_index)

        # Solve least squares problem to find the best coefficients
        subdictionary = D[:, support].conj()
        least_squares_solution = np.linalg.lstsq(subdictionary, x, rcond=None)[0]

        # Update the solution vector
        solution[support] = abs(least_squares_solution.flatten())

        # Update the residual
        residual = x - subdictionary @ least_squares_solution
    solution[solution>0.1]=1 
    return solution

def randSparse(signal,sparsity=0.3):
    sparseSignal = signal.clone()
    sparseInd = torch.randperm(signal.shape[1]-1)[:int(signal.shape[1] * sparsity)] + 1
    sparseSignal[:,sparseInd,:,:] = 0
    return sparseSignal

def randSparse_eval(signal,sparsity=0.3):
    sparseSignal = signal.clone()
    sparseInd = torch.randperm(signal.shape[0]-1)[:int(signal.shape[0] * sparsity)] + 1
    sparseSignal[sparseInd,:] = 0
    return sparseSignal

def normalization_inp(A):
    tmp = A[:, 0, :].unsqueeze(1)
    A_new = A / tmp * torch.abs(tmp)
    return torch.view_as_real(A_new)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        # Calculate the Euclidean distance between the pairs
        dist = F.pairwise_distance(x1, x2)

        # Compute the loss for similar and dissimilar pairs
        loss = (label) * torch.pow(dist, 2) + \
               (1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        
        # Return the mean of the loss over all pairs
        return torch.mean(loss)
    
def main():
    train_loader = create_dataloader(os.path.join('../data', 'val'), batch_size=8)
    for signals1, signals2, labels, trt in (train_loader):
        signals = torch.cat((signals1,signals2),dim=2)
        print(signals.shape)
if __name__ == "__main__":
    main()