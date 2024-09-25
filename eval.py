import argparse
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from scr.helpers import *
from models.SSNDCNet import SSNDCNet
import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, threshold):
    # Binarize predictions based on the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='macro')
    recall = recall_score(y_true, y_pred_binary, average='macro')
    f1 = f1_score(y_true, y_pred_binary, average='macro')
    auc_roc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }

def initialize_metrics(snr_levels):
    return {
        'Accuracy': np.zeros(snr_levels.shape),
        'Precision': np.zeros(snr_levels.shape),
        'F1-Score': np.zeros(snr_levels.shape),
        'Recall': np.zeros(snr_levels.shape),
        'AUC-ROC': np.zeros(snr_levels.shape)
    }

def update_metrics(metrics, results, idx):
    metrics['Accuracy'][idx] = results['Accuracy']
    metrics['Precision'][idx] = results['Precision']
    metrics['F1-Score'][idx] = results['F1-Score']
    metrics['Recall'][idx] = results['Recall']
    metrics['AUC-ROC'][idx] = results['AUC-ROC']

def plt_metrics(snr, metrics,title):
    plt.plot(snr, metrics['f0'],label='CS-OMP')
    plt.plot(snr, metrics['f1'],label='BaseNet1')
    plt.plot(snr, metrics['f2'],label='BaseNet2')
    plt.plot(snr, metrics['f3'],label='Proposed')
    plt.title(title)
    plt.legend()
    plt.show()
    
def save_metrics_data(metrics):
    Accuracy = {}
    Precision = {}
    F1 = {}
    Recall = {}
    AUC = {}
    for idx in range(len(metrics)):
        Accuracy[f'f{idx}'] = metrics[idx]['Accuracy']
        Precision[f'f{idx}'] = metrics[idx]['Precision']
        F1[f'f{idx}'] = metrics[idx]['F1-Score']
        Recall[f'f{idx}'] = metrics[idx]['Recall']  
        AUC[f'f{idx}'] = metrics[idx]['AUC-ROC']           
    
    os.makedirs('./mat', exist_ok=True)     
    savemat('./mat/Accuracy.mat', Accuracy)
    savemat('./mat/Precision.mat', Precision)
    savemat('./mat/F1.mat', F1)
    savemat('./mat/Recall.mat', Recall)
    savemat('./mat/AUC.mat', AUC)
    
    return Accuracy,Precision,F1,Recall,AUC
        
def process_model(model, signals, output_list):
    # Obtain model outputs
    _, _, outputs = model(signals)
    # Process outputs
    outputs = outputs.squeeze(0).detach().cpu().numpy()
    # Append processed outputs to the list
    output_list.append(outputs)
    return output_list
    
def evaluate_model(models, angles, snr_db, args, device='cuda'):
    A = steering_vector(args.number_elements, angles).conj()
    D = A.resolve_conj().numpy()
    y_true = []
    y_cs = []
    y_pred0 = []
    y_pred1 = []
    y_pred2 = []
    for i in range(args.num_simulations):
        deg_indices = torch.randperm(len(angles))[:args.num_targets]
        degs = angles[deg_indices]
        amps = torch.cat((0.5 + 0.5 * torch.rand(args.num_targets - 1), torch.tensor([1.0])))            
        labels = generate_label(angles, deg_indices)
        noisy_signal = generate_complex_signal(args.number_elements, snr_db=snr_db, deg=degs, amp=amps)
        noisy_signal = randSparse_eval(noisy_signal, args.sparsity)
        signals = noisy_signal.repeat(1, 2).to(device).unsqueeze(0)
        signals = normalization_inp(signals)
        # CS-OMP
        x = torch.view_as_complex(signals[0,:,1,:]).cpu().numpy()     
        output_cs = omp(D, x, sparsity_level=args.num_targets) 
        y_cs.append(output_cs)
        y_true.append(labels.cpu().numpy())
        
        y_pred0 = process_model(models[0], signals, y_pred0)
        y_pred1 = process_model(models[1], signals, y_pred1)
        y_pred2 = process_model(models[2], signals, y_pred2)
        
    y_true = np.concatenate(y_true, axis=0)
    y_cs = np.concatenate(y_cs, axis=0)
    y_pred0 = np.concatenate(y_pred0, axis=0)
    y_pred1 = np.concatenate(y_pred1, axis=0)
    y_pred2 = np.concatenate(y_pred2, axis=0)
    
    # Calculate metrics for each model
    metrics_cs = calculate_metrics(y_true, y_cs, args.threshold)
    metrics0 = calculate_metrics(y_true, y_pred0, args.threshold)
    metrics1 = calculate_metrics(y_true, y_pred1, args.threshold)
    metrics2 = calculate_metrics(y_true, y_pred2, args.threshold)    
 
    return metrics_cs, metrics0, metrics1, metrics2

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = [SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, False),
              SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, True),
              SSNDCNet(args.number_elements, args.output_size, args.embed_dim, args.sparsity, True)]

    if torch.cuda.device_count() >= 4:
        models = [nn.DataParallel(model, device_ids=list(range(4))) for model in models]
    models = [model.to(device) for model in models]

    paths = [os.path.join(args.checkpoint_path, subdir, 'best_model_checkpoint.pth') for subdir in ['filled', 'sparse', 'siamese']]
    for model, path in zip(models, paths):
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()
  
    snr_levels = np.arange(0, 31, 1)  # SNR levels from -5 dB to 30 dB in 5 dB steps
    angles = torch.arange(-60, 61, 1)

    metrics = [initialize_metrics(snr_levels) for _ in range(4)]
    
    for cnt, snr_db in enumerate(tqdm(snr_levels, desc='SNR levels', unit='snr')):
        results = [evaluate_model(models, angles, snr_db, args, device=device)]
        for metric, result in zip(metrics, results[0]):
            update_metrics(metric, result, cnt)
           
    Accuracy,Precision,F1,Recall,AUC = save_metrics_data(metrics)  
    plt_metrics(snr_levels, Accuracy,'Accuracy')
    plt_metrics(snr_levels, Precision,'Precision')
    plt_metrics(snr_levels, F1,'F1')
    plt_metrics(snr_levels, Recall,'Recall')
    plt_metrics(snr_levels, AUC,'AUC')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DOA estimation model")
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
    parser.add_argument('--num_simulations', type=int, default=50,
                        help='Number of evaluation signals for each SNR')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold')
    parser.add_argument('--num_targets', type=int, default=3,
                        help='Number of targets')
    args = parser.parse_args()
    main(args)
