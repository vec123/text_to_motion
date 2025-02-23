
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FuncAnimation
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import torch
import motion_models
import utils
import losses
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# Paths to folders
motion_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/new_joints/new_joints"  # Replace with the path to the 'new_joints' folder
text_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/texts/texts"         # Replace with the path to the 'texts' folder
train_file_names = "../KIT-ML-20250125T102306Z-001/KIT-ML/train.txt" 

train_motions, train_descriptions =  utils.load_matching_files(motion_folder, text_folder, train_file_names, short = True)
train_motions, motions_attention_mask = utils.pad_sequences_and_create_mask(train_motions)


#--------------Hyperparams
max_text_length = 800
max_motion_length = train_motions.shape[1]
device = "cpu"

batch_size = 2
encoded_dim = 768
motion_feature_dim = 63
embed_size = 8
num_layers = 8
num_latents = 3
dim_latent = 2
heads = 8
forward_expansion = 1
dropout = 0.1


#--------------Models

modelpath = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(modelpath)
bert_model = AutoModel.from_pretrained(modelpath)


text_encoder = motion_models.Text_encoder_transformers(
         encoded_dim,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout)

motion_encoder= motion_models.Motion_encoder_transformers(
         motion_feature_dim,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
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



#--------------Train

# Create a DataLoader for batching
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Define training parameters
epochs = 1  # Set desired epochs
save_epochs = 1  # Save models and plot every 'save_epochs' epochs

# Custom dataset to tokenize descriptions on-the-fly
class MotionDataset(Dataset):
    def __init__(self, descriptions, motions, masks):
        self.descriptions = descriptions
        self.motions = motions
        self.masks = masks

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.descriptions[idx], self.motions[idx], self.masks[idx]

# Create dataset and DataLoader
dataset = MotionDataset(train_descriptions, train_motions, motions_attention_mask)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Optimizers (Assuming Adam)
optimizer_text = torch.optim.Adam(text_encoder.parameters(), lr=1e-4)
optimizer_motion = torch.optim.Adam(motion_encoder.parameters(), lr=1e-4)
optimizer_decoder = torch.optim.Adam(motion_decoder.parameters(), lr=1e-4)

total_losses = []

for epoch in range(epochs):
    print("epoch: ", epoch)
    for train_descriptions_batch, train_motion_batch, motions_attention_mask_batch in dataloader:
        optimizer_text.zero_grad()
        optimizer_motion.zero_grad()
        optimizer_decoder.zero_grad()

        print("train_motion_batch.shape: ", train_motion_batch.shape)
        # Tokenize descriptions dynamically
        inputs = bert_tokenizer(train_descriptions_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
                bert_embeddings = bert_model(**inputs).last_hidden_state.float()
                bert_embeddings_mask = inputs.attention_mask.to(dtype=bool)
        text_embeddings = bert_embeddings / bert_embeddings.norm(dim=-1, keepdim=True)
        print("text_embeddings.shape: ",text_embeddings.shape)
        mu_text, var_text = text_encoder.forward(text_embeddings, bert_embeddings_mask)
        print("mu_text.shape: ",mu_text.shape)

        train_motion_batch = train_motion_batch.view(train_motion_batch.shape[0], train_motion_batch.shape[1], -1)
        mu_motion, var_motion = motion_encoder.forward(train_motion_batch, motions_attention_mask_batch)
        print("mu_motion.shape: ",mu_motion.shape)

        z_text = utils.reparametrization(mu_text, var_text)
        z_motion = utils.reparametrization(mu_motion, var_motion)
        print("z_text.shape: ",z_text.shape)

        input_pos = torch.tensor(np.arange(0, train_motion_batch.shape[1])).unsqueeze(0).unsqueeze(2)
        input_pos = input_pos.expand(batch_size, train_motion_batch.shape[1], 1)
        print("input_pos.shape:", input_pos.shape)
        motion_from_text = motion_decoder.forward(input_pos, z_text, None, None)
        motion_from_motion = motion_decoder.forward(input_pos, z_motion, None, None)
        print("motion_from_text.shape:", motion_from_text.shape)
        loss = losses.train_loss(
            motion_from_text, 
            motion_from_motion, 
            train_motion_batch, 
            z_text, 
            z_motion,
            mu_text, 
            var_text,
            mu_motion, 
            var_motion
        )
        
        loss.backward()
        optimizer_text.step()
        optimizer_motion.step()
        optimizer_decoder.step()
        
        total_losses.append(loss.item())
        print("loss:", loss)
        break
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Save models and plot every 'save_epochs' epochs
    if (epoch + 1) % save_epochs == 0:
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(text_encoder.state_dict(), model_dir + f"/text_encoder_epoch_{epoch+1}.pth")
        torch.save(motion_encoder.state_dict(), model_dir + f"/motion_encoder_epoch_{epoch+1}.pth")
        torch.save(motion_decoder.state_dict(), model_dir + f"/motion_decoder_epoch_{epoch+1}.pth")
        
        # Plot total loss
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(total_losses) + 1), total_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Over Time")
        plt.savefig(plot_dir + f"/training_loss_epoch_{epoch+1}.png")
        plt.close()
