
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FuncAnimation
from transformers import CLIPTokenizer, CLIPModel, BertTokenizer, BertModel
import torch
import motion_models
import utils
import losses



# Paths to folders
motion_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/new_joints/new_joints"  # Replace with the path to the 'new_joints' folder
text_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/texts/texts"         # Replace with the path to the 'texts' folder
train_file_names = "../KIT-ML-20250125T102306Z-001/KIT-ML/train.txt" 

train_motions, train_descriptions =  utils.load_matching_files(motion_folder, text_folder, train_file_names, short = True)
train_motions, motions_attention_mask = utils.pad_sequences_and_create_mask(train_motions)


#-----------Hyperparams

batch_size = 2
encoded_dim = 768
motion_feature_dim = 63
embed_size = 16
num_layers = 8
num_latents = 3
dim_latent = 2
heads = 8
forward_expansion = 1
dropout = 0.1
max_text_length = 800
max_motion_length = train_motions.shape[1]
device = "cpu"


#-----------Models

bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

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



#-----------Forward pass

train_descriptions_batch = train_descriptions[0:batch_size]
train_motion_batch = train_motions[0:batch_size]
motions_attention_mask_batch = motions_attention_mask[0:batch_size]



inputs = bert_tokenizer(train_descriptions_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    bert_embeddings = bert_model(**inputs).last_hidden_state
text_embeddings = bert_embeddings
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

mu_text, var_text = text_encoder.forward(text_embeddings, None)

train_motion_batch = train_motion_batch.view(train_motion_batch.shape[0], train_motion_batch.shape[1], -1)
mu_motion, var_motion = motion_encoder.forward(train_motion_batch, motions_attention_mask_batch)

z_text = utils.reparametrization(mu_text, var_text)
z_motion = utils.reparametrization(mu_motion, var_motion)

input_pos = torch.tensor( np.arange(0,train_motion_batch.shape[1]) ).unsqueeze(0).unsqueeze(2)
input_pos = input_pos.expand(batch_size, train_motion_batch.shape[1], 1)

motion_from_text = motion_decoder.forward(input_pos, z_text, None, None)
motion_from_motion = motion_decoder.forward(input_pos, z_motion, None, None)

loss = losses.train_loss(
    motion_from_text,  motion_from_motion, train_motion_batch, z_text, z_motion, mu_text, var_text, mu_motion,  var_motion
    )

print("loss.shape", loss.shape)
print(loss)
  