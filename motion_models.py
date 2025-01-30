
import transformer_modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Motion_encoder_transformers(nn.Module):
    def __init__(self, 
         src_vocab_size,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout,
         max_length):
       
       super(Motion_encoder_transformers, self).__init__()
       self.device = device
       self.mean_encoder = transformer_modules.Motion_Encoder(
         src_vocab_size,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout,
         max_length)
       self.logvar_encoder = transformer_modules.Motion_Encoder(
         src_vocab_size,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout,
         max_length)
    
    def forward(self, motion, motion_mask):
        out_mean = self.mean_encoder(motion, motion_mask)
        out_logvar = self.logvar_encoder(motion, motion_mask)
       
        return out_mean, out_logvar

class Text_encoder_transformers(nn.Module):
    def __init__(self, 
         encoded_dim,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout
        ):
       super(Text_encoder_transformers, self).__init__()
       self.mean_encoder = transformer_modules.Text_Encoder(
         encoded_dim,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout)
       
       self.logvar_encoder = transformer_modules.Text_Encoder(
         encoded_dim,
         embed_size,
         dim_latent,
         num_latents,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout)
    

    def forward(self, text_embeddings, text_mask):
        out_mean = self.mean_encoder(text_embeddings, text_mask)
        out_logvar = self.logvar_encoder(text_embeddings, text_mask)
        return out_mean, out_logvar

class Motion_decoder_transformers(nn.Module):

    def __init__(
         self,
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
   ):
      super(Motion_decoder_transformers, self).__init__()
      self.motion_decoder = transformer_modules.Motion_Decoder(  
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
      
    def forward(self, x, z, src_mask, trg_mask):
       out = self.motion_decoder(x,z, src_mask, trg_mask)
       return out



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)

    trg = torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)

    seq_len, feature_dim = x.shape
    src_pad_idx = 0
    trg_pad_idx = 0
    text_vocab_size = 10
    motion_embed_size = 10
    embed_size = 120
    num_layers = 6
    num_latents = 3
    dim_latents = 2
    heads = 6
    forward_expansion = 1
    dropout = 0
    max_length = 500

    text_encoder = Text_encoder_transformers(
         text_vocab_size,
         embed_size,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout,
         max_length)
    
    encoded_text_mean, encoded_text_logvar = text_encoder.forward(x, None)
  
    motion_encoder= Motion_encoder_transformers(
         motion_embed_size,
         embed_size,
         num_layers,
         num_latents,
         dim_latents,
         heads,
         device,
         forward_expansion,
         dropout,
         max_length)
    
    encoded_motion_mean, encoded_motion_logvar = motion_encoder.forward(x, None)
    print(encoded_text_mean.shape, encoded_text_logvar.shape)

    motion_decoder= Motion_decoder_transformers(
         latent_num,
         latent_dim,
         embed_size,
         feature_dim,
         num_layers,
         heads,
         forward_expansion,
         dropout,
         device,
         max_length
         )
    print("---------")
    z = torch.randn(2, embed_size) 
    out = motion_decoder.forward(z , 12, None, None)
    print(out.shape)