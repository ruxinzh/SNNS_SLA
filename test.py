import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from scr.helpers import *
from models.SSNDCNet import SSNDCNet
import matplotlib.pyplot as plt

def randSparse(signal,sparsity):
    sparseSignal = signal.clone()
    sparseInd = torch.randperm(signal.numel()-1)[:int(signal.numel() * sparsity)] + 1
    sparseSignal[sparseInd] = 0
    return sparseSignal
def randSparse2(signal1,signal2,sparsity):
    sparseSignal1 = signal1.clone()
    sparseSignal2 = signal2.clone()
    sparseInd1 = torch.randperm(signal1.numel()-1)[:int(signal1.numel() * sparsity)] + 1
    sparseInd2 = torch.randperm(signal1.numel()-1)[:int(signal1.numel() * sparsity)] + 1
    sparseSignal1[sparseInd1] = 0
    sparseSignal2[sparseInd2] = 0
    return sparseSignal1,sparseSignal2
def normalization_inp(A):
    tmp = A[:, 0, :].unsqueeze(1)
    A_new = A / tmp * torch.abs(tmp)
    return torch.view_as_real(A_new)
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, args.use_sparse)

    # Check if multiple GPUs are available and wrap the model using DataParallel
    if torch.cuda.device_count() >= 4:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(4)))  # Modify here to change the number of GPUs used
    model = model.to(device)
    
    # Determine the directory based on conditions
    if args.use_sparse and args.use_siamese:
        subfolder = 'siamese'
    elif args.use_sparse:
        subfolder = 'sparse'
    else:
        subfolder = 'filled'
    # Construct the full checkpoint path
    save_checkpoint_path = os.path.join(args.checkpoint_path, subfolder)
    os.makedirs(save_checkpoint_path, exist_ok=True) 
                
    checkpoint_path = os.path.join(save_checkpoint_path, 'best_model_checkpoint.pth')
    model.load_state_dict(torch.load(checkpoint_path),strict= False)
    
    # Training loop
    best_val_loss = float('inf')
    
    angles = torch.arange(-60, 61, 1)
    A = steering_vector(args.number_elements, angles)
    AH = steering_vector(args.number_elements, angles).conj().T

    model.eval()
    max_targets = 2
    num_targets = torch.randint(1, max_targets + 1, (1,)).item()
    deg_indices = torch.randperm(len(angles))[:num_targets]
    snr_db = 30
    threshold = 0.5
    
    degs = angles[deg_indices]
    amps = torch.cat((0.5 + 0.5 * torch.rand(num_targets - 1), torch.tensor([1.0])))                     
    label = generate_label(angles,deg_indices)
    noisy_signal = generate_complex_signal(N=args.number_elements, snr_db=snr_db, deg=degs, amp=amps)
    noisy_signal = randSparse_eval(noisy_signal, args.sparsity)
    signals = noisy_signal.repeat(1, 2).to(device).unsqueeze(0)
    signals = normalization_inp(signals)
    _,_,out = model(signals)
    out = out[0,:].cpu().detach().numpy()
    out_binary = (out >= threshold)

    A = steering_vector(args.number_elements, angles).conj()
    D = A.resolve_conj().numpy()
    x = torch.view_as_complex(signals[0,:,1,:]).cpu().numpy()     
    output_cs = omp(D, x, sparsity_level=num_targets)  
    
    plt.figure(figsize=(8, 6))
    plt.plot(angles.numpy(),out_binary,label='Pred')
    # plt.plot(angles.numpy(),output_cs,label='CS-OMP')
    arr = degs.numpy()
    for value in arr:
        plt.axvline(x=value, color='r', linestyle='--')
    plt.legend()
    plt.show()
    print(degs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DOA estimation model")
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to training and validation data directory')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint',
                        help='Path where to save model checkpoints')
    parser.add_argument('--number_elements', type=int, default=20,
                        help='Number of array elements in the model')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='size of embedding dimension')
    parser.add_argument('--output_size', type=int, default=121,
                        help='Output size of the model')
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help='Sparsity level used in the model')
    parser.add_argument('--use_sparse', type=bool, default=True,
                        help='Whether to use sparse augmentation layer in the model')
    parser.add_argument('--use_siamese', type=bool, default=True,
                        help='Whether to use siamese structure')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Whether to load pretrained weights')    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    args = parser.parse_args()
    main(args)
