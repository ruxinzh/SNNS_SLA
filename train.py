import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from scr.helpers import *
from models.SSNDCNet import SSNDCNet


def validate_model(model, dataloader, criterion, device):
    """Perform validation and return the average loss."""
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for signals1, signals2, labels, trt in dataloader:
            signals = torch.cat((signals1,signals2),dim=2)
            signals = signals.to(device).squeeze(-1)
            labels = labels.to(device)
            signals = normalization_inp(signals)
            f1,f2,out = model(signals)
            loss2 = criterion(out,labels.to(out.dtype))
            total_loss = loss2
            val_loss += total_loss.item()
    return val_loss / len(dataloader)

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
    
    # Loss function and optimizer
    margin = 1.0  # This is the margin that is enforced between positive and negative pairs
    criterion1 = ContrastiveLoss(margin)
    criterion2 = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_loader = create_dataloader(os.path.join(args.data_path, 'train'), batch_size=args.batch_size)
    val_loader = create_dataloader(os.path.join(args.data_path, 'val'), batch_size=args.batch_size)

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
    final_model_path = os.path.join(save_checkpoint_path, 'final_model.pth')
    if args.pretrain and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path),strict= False)
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for signals1, signals2, labels, trt in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]"):
            signals = torch.cat((signals1,signals2),dim=2)
            signals = signals.to(device)
            labels = labels.to(device)
            trt = trt.to(device) 
            f1,f2,out = model(normalization_inp(signals))
            loss1 = criterion1(f1, f2, trt)
            loss2 = criterion2(out,labels.to(out.dtype))
            if args.use_siamese:
                total_loss = loss1+loss2
            else:
                total_loss = loss2
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        average_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch + 1} Training Loss: {average_train_loss}")

        if (epoch + 1) % 5 == 0:
            val_loss = validate_model(model, val_loader, criterion2, device)
            print(f"Epoch {epoch + 1} Validation Loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved with validation loss {best_val_loss} at {checkpoint_path}")

    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

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
    parser.add_argument('--use_sparse', type=bool, default=False,
                        help='Whether to use sparse augmentation layer in the model')
    parser.add_argument('--use_siamese', type=bool, default=False,
                        help='Whether to use siamese structure')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Whether to load pretrained weights')    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train the model')

    args = parser.parse_args()
    main(args)
