import argparse
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from scr.helpers import *
from models.SSNDCNet import SSNDCNet
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.decomposition import PCA


def evaluate_model(model0,model1, model2,num_simulations,sparsity, device='cuda'):
    angles = torch.arange(-60, 61, 1)
    all_labels_comb = generate_combinations(121, 3)
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for easy visualization
    classes_idx = torch.randperm(len(all_labels_comb))[0:1]
    # classes_idx = torch.tensor([120223])
    # print(classes_idx)
    pca_f0 = []
    pca_f1 = []
    pca_f2 = []
    for cidx in range(len(classes_idx)):
        idx = classes_idx[cidx]
        label = all_labels_comb[idx]
        label = torch.tensor(label)
        deg_indices = torch.where(label)[0]
        degs = angles[deg_indices]
        num_targets = deg_indices.shape[0] 
        signal_all = []
        for i in tqdm(range(num_simulations)):
            snr_db = 10+30 * torch.rand(1)
            amp = 0.5 + 0.5 * torch.rand(num_targets-1)
            amps = torch.cat((amp, torch.tensor([1.0])))            
            noisy_signal_1 = generate_complex_signal(N=20, snr_db=snr_db, deg=degs, amp=amps)
            noisy_signal_1 = randSparse_eval(noisy_signal_1, sparsity)
            signals = torch.cat((noisy_signal_1, noisy_signal_1), dim=1)
            signals = signals.to(device).unsqueeze(0)
            signal_all.append(signals)
        
        signals = torch.cat(signal_all,dim=0)
        signals = normalization_inp(signals) 
        f01,f2,outputs0 = model0(signals)            
        f11,f2,outputs1 = model1(signals)
        f21,f2,outputs2 = model2(signals)
        features0_np = f01.cpu().detach().numpy()
        features1_np = f11.cpu().detach().numpy()
        features2_np = f21.cpu().detach().numpy()

        reduced_features0 = pca.fit_transform(features0_np)
        reduced_features1 = pca.fit_transform(features1_np)
        reduced_features2 = pca.fit_transform(features2_np)
        
        pca_f0.append(reduced_features0)  
        pca_f1.append(reduced_features1)  
        pca_f2.append(reduced_features2)  


        # Create some example data
        data = {
            'f0': pca_f0,  # Example matrix
            'f1': pca_f1,  # Another matrix
            'f2': pca_f2  # Another matrix
        }
        
        # Save this data to a .mat file
        if sparsity != 0:
            savemat('./mat/feature_sla.mat', data)    
        else :
            savemat('./mat/feature_ula.mat', data)
    return pca_f0, pca_f1, pca_f2


def plot_pca_results(reduced_features, title='ULA'):
    """
    Plot PCA results with different colors for each class.

    Parameters:
    reduced_features (list of arrays): A list where each element is an array of reduced features for a class.
    class_labels (list of str): Labels for each class corresponding to the arrays in reduced_features.
    """
    class_labels = [f'Class {i}' for i in range(len(reduced_features))]
    plt.figure(figsize=(8, 6))
    
    # Iterate over each class's features and labels
    for features, label in zip(reduced_features, class_labels):
        plt.scatter(features[:, 0], features[:, 1], alpha=0.6, label=label)
    
    # Setting plot title and labels
    plt.title(title+':PCA Result of Embedded Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Setting x and y limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # Adding a legend
    plt.legend()
    
    # Display the plot
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model0 = SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, False)
    model1 = SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, True)
    model2 = SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, True)
    
    # Check if multiple GPUs are available and wrap the model using DataParallel
    if torch.cuda.device_count() >= 4:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model0 = nn.DataParallel(model0, device_ids=list(range(4)))  # Modify here to change the number of GPUs used
        model1 = nn.DataParallel(model1, device_ids=list(range(4)))  # Modify here to change the number of GPUs used
        model2 = nn.DataParallel(model2, device_ids=list(range(4)))  # Modify here to change the number of GPUs used

    model0 = model0.to(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    
    save_checkpoint_path0 = os.path.join(args.checkpoint_path, 'filled')
    save_checkpoint_path1 = os.path.join(args.checkpoint_path, 'sparse')
    save_checkpoint_path2 = os.path.join(args.checkpoint_path, 'siamese')
    
    checkpoint_path0 = os.path.join(save_checkpoint_path0, 'best_model_checkpoint.pth')
    checkpoint_path1 = os.path.join(save_checkpoint_path1, 'best_model_checkpoint.pth')
    checkpoint_path2 = os.path.join(save_checkpoint_path2, 'best_model_checkpoint.pth')

    model0.load_state_dict(torch.load(checkpoint_path0),strict= False)
    model1.load_state_dict(torch.load(checkpoint_path1),strict= False)
    model2.load_state_dict(torch.load(checkpoint_path2),strict= False)
      
    model0.eval()
    model1.eval()
    model2.eval()
    num_simulations = 5000
    pca0,pca1,pca2 = evaluate_model(model0, model1, model2, num_simulations, sparsity = args.sparsity, device=device)
    plot_pca_results(pca0,'ULA')
    plot_pca_results(pca1,'SLA1')
    plot_pca_results(pca2,'SLA2')
    

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
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train the model')

    args = parser.parse_args()
    main(args)
