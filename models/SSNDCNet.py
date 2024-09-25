import torch
import torch.nn as nn
import sys
sys.path.append('../')
from scr.helpers import *
import torch.nn.functional as F

import os
from scr.helpers import * 
import matplotlib.pyplot as plt

class SparseLayer(nn.Module):
    def __init__(self, input_size=10, max_sparsity=0.5):
        super(SparseLayer, self).__init__()
        self.input_size = input_size
        self.max_zeros = int(input_size * max_sparsity)

    def forward(self, x):
        if self.training:
            batch_size,N,Ns = x.size()  # Get the batch size
            sparsity = torch.zeros(batch_size, dtype=torch.long).to(x.device)  # Tensor to store the number of zeros
            # Generate a random mask for each example in the batch
            masks1 = torch.ones((batch_size, self.input_size)).to(x.device)  # Start with all ones
            masks2 = torch.ones((batch_size, self.input_size)).to(x.device)  # Start with all ones

            num_zeros = torch.randint(0, self.max_zeros + 1, (1,))
            zero_indices1 = torch.randperm(self.input_size)[:num_zeros]  # Random indices to be zeroed
            zero_indices2 = torch.randperm(self.input_size)[:num_zeros]  # Random indices to be zeroed

            masks1[:, zero_indices1] = 0  # Set selected indices to zero
            masks2[:, zero_indices2] = 0  # Set selected indices to zero

            sparsity = N - num_zeros
            masks = torch.cat((masks1.unsqueeze(2),masks2.unsqueeze(2)), dim=2)
            x_sparse = x * masks

        else:
            x_sparse = x
            sparsity, masks = self.thresholding(x)

        return x_sparse, sparsity.to(x_sparse.device), masks.to(x_sparse.device)


    def thresholding(self, x):
        threshold = 0.001
        mask = (torch.abs(x) > threshold).float()
        return torch.sum(mask, dim=1)[0,0], mask
    

class SALayer(nn.Module):
    def __init__(self, number_element=10, output_size=2048, max_sparsity=0.5, angles = torch.arange(-60, 60.5, 0.5)):
        super(SALayer, self).__init__()
        # Initialize SparseLayer
        self.sparselayer = SparseLayer(number_element, max_sparsity)
        self.angles = angles
        self.AH = steering_vector(number_element, self.angles).conj().T
        self.output_size = output_size
        self.sigEncode = nn.Linear(number_element*2,output_size)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x, is_sparse):
        batch_size, N, _ = x.size()
        # Sparse representation 
        if is_sparse:
            x_sparse, sparsity, masks = self.sparselayer(x)
            sig1 = x_sparse[:,:,0]
            sig2 = x_sparse[:,:,1]
            
            sig1_encode = self.sigEncode(torch.view_as_real(sig1).reshape(batch_size,-1))/sparsity
            sig2_encode = self.sigEncode(torch.view_as_real(sig2).reshape(batch_size,-1))/sparsity

            sig1_encode = self.relu(sig1_encode)
            sig2_encode = self.relu(sig2_encode)
            sig1_FFT = torch.view_as_real(torch.matmul(self.AH.to(sig1.device),sig1.T).T/sparsity).permute(0,2,1)
            sig2_FFT = torch.view_as_real(torch.matmul(self.AH.to(sig2.device),sig2.T).T/sparsity).permute(0,2,1)

        else:
            sig1 = x[:,:,0]
            sig2 = x[:,:,1]

            sig1_encode = self.sigEncode(torch.view_as_real(sig1).reshape(batch_size,-1))/N
            sig2_encode = self.sigEncode(torch.view_as_real(sig2).reshape(batch_size,-1))/N

            sig1_encode = self.relu(sig1_encode)
            sig2_encode = self.relu(sig2_encode)

            sig1_FFT = torch.view_as_real(torch.matmul(self.AH.to(sig1.device),sig1.T).T/N).permute(0,2,1)
            sig2_FFT = torch.view_as_real(torch.matmul(self.AH.to(sig2.device),sig2.T).T/N).permute(0,2,1)

        return sig1_encode, sig1_FFT, sig2_encode, sig2_FFT

               
class SSNDCNet(nn.Module):
    def __init__(self, number_element=10, output_size=15, hidden_dim = 2048, max_sparsity=0.3, is_sparse=True):
        super(SSNDCNet, self).__init__()
        self.angles = torch.arange(-60, 61,1)
        self.salayer = SALayer(number_element, hidden_dim, max_sparsity,self.angles)
        self.is_sparse = is_sparse
        self.flat = nn.Flatten(1, 2)     
        # self.A = steering_vector(output_size, self.angles)
        # Encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # Adding MaxPooling
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # Adding MaxPooling
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)  # Adding MaxPooling
        )        
        
        self.fc_encoder0 = nn.Sequential(nn.Linear(32*15+128, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128)
                        )       
        self.sig_encoder = nn.Sequential(nn.Linear(512, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128) ,
                        nn.ReLU()
                        ) 
        self.cla = nn.Sequential(nn.Sigmoid(),
                        nn.Linear(128, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, output_size),
                        nn.Sigmoid()
                        ) 
    def forward(self, x):
        if len(x.size()) == 4: # signal pass in as real value
            x  = torch.view_as_complex(x)
            
        sig1,sig1_fft,sig2,sig2_fft = self.salayer(x,self.is_sparse)
        
        fft1_feature = self.flat(self.freq_encoder(sig1_fft))       
        fft2_feature = self.flat(self.freq_encoder(sig2_fft))
        
        sig1_feature = self.sig_encoder(sig1)
        sig2_feature = self.sig_encoder(sig2) 

        feature1 = torch.cat((sig1_feature,fft1_feature),dim=1)
        feature2 = torch.cat((sig2_feature,fft2_feature),dim=1)
        
        feature1 = self.fc_encoder0(feature1)
        feature2 = self.fc_encoder0(feature2)
        
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        
        output = self.cla(feature1)
        return feature1, feature2, output
      