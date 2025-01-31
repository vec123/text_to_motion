
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FuncAnimation
from transformers import CLIPTokenizer, CLIPModel
import torch
import motion_models
import utils

def load_text(file_path):
    """
    Loads and prints the text description.
    """
    with open(file_path, 'r') as f:
        text = f.read().strip()
    return text


def load_matching_files(motion_folder, text_folder, matching_file):
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
            motions.append(torch.tensor(motion))
            descriptions.append(description)
        else:
            print(f"Warning: Missing file for {file_name} in {motion_folder} or {text_folder}")

    return motions, descriptions


# Paths to folders
motion_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/new_joints/new_joints"  # Replace with the path to the 'new_joints' folder
text_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/texts/texts"         # Replace with the path to the 'texts' folder
train_file_names = "../KIT-ML-20250125T102306Z-001/KIT-ML/train.txt" 

train_motions, train_descriptions = load_matching_files(motion_folder, text_folder, train_file_names)
print("train_motions: ", len(train_motions))
print("train_motions[0]: ",train_motions[0].shape)
train_motions, motions_attention_mask = utils.pad_sequences_and_create_mask(train_motions)
print("train_motions: ",train_motions.shape)
print("train_motions[0]: ",train_motions[0].shape)

# Example: Choose a specific motion
motion_file = os.path.join(motion_folder, "M02574.npy")  # Replace with the motion file name
text_file = os.path.join(text_folder, "M02574.txt")      # Replace with the text file name

# Load motion and text
np_motion = np.load(motion_file)
np_motion = np_motion.reshape(np_motion.shape[0],-1 )  
txt_description = load_text(text_file)
motion_tensor = torch.from_numpy(np_motion).unsqueeze(0)


model_name = "openai/clip-vit-base-patch32"  
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Tokenize the input string
inputs = tokenizer([txt_description], padding=True, return_tensors="pt")
# Encode the string into an embedding
with torch.no_grad():  # Disable gradient computation for inference
    text_embeddings = model.get_text_features(**inputs)

# Normalize embeddings (optional)
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
# Print the embedding shape and values
print("text embedding shape:", text_embeddings.shape)
print("Motion tensor shape:", motion_tensor.shape)

motion_feature_dim = np_motion.shape[1]
embed_size = 16
num_layers = 8
num_latents = 331
dim_latent = 2
heads = 8
forward_expansion = 1
dropout = 0.1
max_text_length = 800
max_motion_length = motion_tensor.shape[1]
device = "cpu"

text_encoder = motion_models.Text_encoder_transformers(
         text_embeddings.shape[1],
         embed_size,
         num_layers,
         num_latents,
         dim_latent,
         heads,
         device,
         forward_expansion,
         dropout)

motion_encoder= motion_models.Motion_encoder_transformers(
         motion_feature_dim,
         embed_size,
         num_layers,
         num_latents,
         dim_latent,
         heads,
         device,
         forward_expansion,
         dropout,
         max_motion_length)

motion_decoder = motion_models.Motion_decoder_transformers( 
         motion_feature_dim,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout,
         max_motion_length
         )


encoded_text_mean, encoded_text_logvar = text_encoder.forward(text_embeddings, None)
print("encoded_text_mean.shape", encoded_text_mean.shape)
print("encoded_text_logvar.shape", encoded_text_logvar.shape)


encoded_motion_mean, encoded_motion_logvar = motion_encoder.forward(motion_tensor, None)
print("encoded_motion_mean.shape", encoded_motion_mean.shape)
print("encoded_motion_logvar.shape", encoded_motion_logvar.shape)

input_pos = torch.tensor( np.arange(0,np_motion.shape[0]) ).unsqueeze(0).unsqueeze(2)
print("input_pos.shape", input_pos.shape)
generated_motion = motion_decoder.forward(input_pos, encoded_motion_mean, None, None)
print("generated_motion.shape", generated_motion.shape)
  