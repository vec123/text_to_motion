from typing import List, Dict
import torch
from torch import Tensor
import numpy as np
import torch
from torch import nn
import os
import torch.nn.functional as F


def load_text(file_path):
    """
    Loads and prints the text description.
    """
    with open(file_path, 'r') as f:
        text = f.read().strip()
    return text

def load_matching_files(motion_folder, text_folder, matching_file, short = False):
    """
    Loads motions and descriptions whose file names match those listed in a text file.
    """
    # Read the file containing the list of file names (without extensions)
    with open(matching_file, 'r') as f:
        matching_names = [line.strip() for line in f.readlines()]

    # Initialize storage for motions and descriptions
    motions = []
    descriptions = []

    for file_name in matching_names:
        motion_file = os.path.join(motion_folder, f"{file_name}.npy")
        text_file = os.path.join(text_folder, f"{file_name}.txt")

        if os.path.exists(motion_file) and os.path.exists(text_file):
            # Load motion and description
            motion = np.load(motion_file)  # Shape: (num_frames, num_joints, 3)
            description = load_text(text_file)

            # Append to lists
            if short:
                if motion.shape[0] < 500:
                    description = load_text(text_file)
                    motions.append(torch.tensor(motion))
                    descriptions.append(description)
            else:
                motions.append(torch.tensor(motion))
                descriptions.append(description)

        else:
            print(f"Warning: Missing file for {file_name} in {motion_folder} or {text_folder}")

    return motions, descriptions

def reparametrization(mu, logvar):
    std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
    eps = torch.randn_like(std)
    return mu + eps * std

def pad_sequences_and_create_mask(sequences: List[Tensor], pad_value=float("-1e20"), device: torch.device = torch.device("cpu")) -> Tensor:
    """
    Pads sequences to the same length and creates an attention mask.
    
    Args:
    - sequences: List of tensors with different sequence lengths
    - pad_value: Value to use for padding
    - device: Device where tensors will be stored (CPU or CUDA)
    
    Returns:
    - padded_sequences: A tensor of shape (batch_size, max_seq_len)
    - attention_mask: A tensor of shape (batch_size, max_seq_len), where 1 represents real tokens, 0 represents padding
    """
    # First, pad the sequences using pad_sequence
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_value).to(device)
    
    # Create attention mask (1 for non-padding, 0 for padding)
    attention_mask = padded_sequences != pad_value
    
     # Create pad_indices (the positions of the padding tokens, where padded sequences are equal to pad_value)
    attention_mask = attention_mask.view(attention_mask.shape[0], attention_mask.shape[1], -1)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = attention_mask[..., 0] 
    return padded_sequences, attention_mask

def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    """
    Converts a list of sequence lengths into an attention mask where padding positions are masked out.
    
    Args:
    - lengths: A list of sequence lengths (int) for each sequence in the batch
    - device: Device where the tensors will be stored (CPU or CUDA)
    
    Returns:
    - mask: A mask tensor (shape: batch_size, max_len), where 1 means a real token and 0 means padding
    """
    # Convert lengths to a tensor
    lengths_tensor = torch.tensor(lengths, device=device)
    
    # Get the maximum sequence length in the batch
    max_len = lengths_tensor.max()
    
    # Generate a tensor of shape (batch_size, max_len)
    mask = torch.arange(max_len, device=device).expand(len(lengths_tensor), max_len) < lengths_tensor.unsqueeze(1)
    return mask

def make_mask(src, pad_idx, device):
    print("src.shape", src.shape)
    print("pad_idx.shape", pad_idx.shape)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)