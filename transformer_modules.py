import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class SelfAttention(nn.Module):
    
    def __init__(self, embed_size, heads):
      
      super(SelfAttention,self).__init__()

      self.embed_size = embed_size
      self.heads = heads
      self.head_dim = embed_size//heads
      
      assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

      self.values = nn.Linear(self.head_dim, self.head_dim, bias= False)
      self.keys = nn.Linear(self.head_dim, self.head_dim, bias= False)
      self.queries = nn.Linear(self.head_dim, self.head_dim, bias= False)

      self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, queries, keys, values, mask):
      N = queries.shape[0]
      value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

      values = values.reshape(N, value_len, self.heads,self.head_dim)
      keys = keys.reshape(N, key_len, self.heads,self.head_dim)
      queries = queries.reshape(N, query_len, self.heads,self.head_dim)

      values = self.values(values)
      keys = self.keys(keys)
      queries = self.queries(queries)

      energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
      #queries shape: (N, query_len, heads, heads_dim)
      #keys shape: (N, key_len, heads, heads_dim)
      #energy shape: (N, heads, query_len, key_len)
      if mask is not None:
         energy = energy.masked_fill(mask == False, float("-1e20") )
      energy = torch.clamp(energy, min=-1e6, max=1e6)
      attention = torch.softmax( energy / (self.embed_size **(1/2)), dim = 3 )

      out = torch.einsum("nhql,nlhd -> nqhd", [attention, values]).reshape(N, query_len,
                                                       self.heads*self.head_dim)
      #attention shape: (N, heads, query_len, key_len) 
      #values shape: (N, value_len, heads, heads_dim)
      #(N, query_len, heads, head_dim)
      #after einsum (N, query_len, heads, head_dim) then flatten the last two dimensions

      out = self.fc_out(out)
      return out
    
class TransformerBlock(nn.Module):
   
   def __init__(self, embed_size, heads, dropout, forward_expansion):
      super(TransformerBlock, self).__init__()

      self.attention = SelfAttention(embed_size, heads)
      self.norm1 = nn.LayerNorm(embed_size, eps=1e-6)
      self.norm2 = nn.LayerNorm(embed_size, eps=1e-6)

      self.feedforward = nn.Sequential(
         nn.Linear(embed_size, forward_expansion*embed_size), 
         nn.ReLU(),
         nn.Linear(forward_expansion*embed_size, embed_size)
      )
      self.dropout = nn.Dropout(dropout)

   def forward(self, value, key, query, mask):
     
      attention = self.attention(value, key, query, mask)
     
      x = attention + query
      x = torch.clamp(x, min=-1e6, max=1e6)  # ✅ Clamp before LayerNorm
      x = self.dropout(self.norm1(x))
     
      forward = self.feedforward(x)
    
      out = self.dropout( self.norm2(forward + x))
      
      return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
      super(DecoderBlock, self).__init__()
      self.attention = SelfAttention(embed_size, heads)
      self.norm = nn.LayerNorm(embed_size)
      self.transformer_block = TransformerBlock(
         embed_size, heads, dropout, forward_expansion
      )
      self.dropout = nn.Dropout(dropout)
      self.device = device 

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, embed_size, n_out, d_out, heads, forward_expansion, dropout, device):
        super(AttentionPooling, self).__init__()
        
        self.n_out = n_out  # Number of latents (output sequence length)
        self.d_out = d_out  # Output embedding dimension
        self.embed_size = embed_size
        self.heads = heads
        self.device = device

        # Learnable query embeddings
        self.latent_queries_n  = nn.Parameter(torch.randn(1, n_out, embed_size) * 0.01)
        #linear layer
        self.projection = nn.Linear(embed_size, d_out)
        # Self-Attention layer to compute interactions
        self.self_attention = SelfAttention(embed_size, heads)
        # Transformer Block for richer representation
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        # Final LayerNorm for stability
        self.norm = nn.LayerNorm(embed_size, eps=1e-6)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length, embed_size = x.shape

        # Expand latent queries to batch size
        Q = self.latent_queries_n.expand(N, -1, -1)  # (N, n_out, embed_size)
        # Multi-head Self-Attention (Attends input sequence using learnable queries)
        x = self.self_attention(Q, x, x, None)  # (N, n_out, embed_size)
        output = self.projection(x)  # (N, n_out, embed_size))
        return output  # (N, n_out, d_out)
       
class AttentionPooling_old(nn.Module):
    def __init__(self, embed_size, n_out, d_out, heads, forward_expansion, dropout, device):
        super(AttentionPooling_old, self).__init__()
        
        self.n_out = n_out
        self.d_out = d_out
        self.embed_size = embed_size
        # Learnable queries (latents) to attend to the input sequence intermediate_dim must be either embed_dimension or sequence_length
        self.latent_queries_latent_dim= nn.Parameter(torch.randn(1, embed_size, d_out))  # (1, num_latents, dim_latent)
        self.latent_queries_latent_nums = nn.Parameter(torch.randn(1, embed_size, n_out))
        self.device = device
        self.norm = nn.LayerNorm(d_out)
    def forward(self, x, mask=None):
         N, seq_length, embed_size = x.shape
       
         K = x  
         V = x  
         Q = self.latent_queries_latent_nums
         attn_scores = torch.einsum('ben,ble->bnl', Q, K)  
         attn_weights = torch.softmax( attn_scores / (self.embed_size **(1/2)), dim = 0 ) #F.softmax(attn_scores, dim=-1)
         context_1 = torch.einsum('bnl,ble->bne', attn_weights, V)  
     
         K = context_1 
         V = context_1
         Q = self.latent_queries_latent_dim
         energy = torch.einsum('bed,bne->bde', Q, K)  

         attn_weights = torch.softmax( energy / (self.embed_size **(1/2)), dim = 0 ) #F.softmax(attn_scores, dim=-1)

         output = torch.einsum('bde,bne->bnd', attn_weights, V)  
       
         return output



class Motion_Encoder(nn.Module):

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
         max_length
    ):
      super(Motion_Encoder, self).__init__()
      self.embed_size = embed_size
      self.device = device
      self.motion_feature_dim = motion_feature_dim
      self.embed_size = embed_size
      self.max_length = max_length
      self.motion_embedding = nn.Linear(motion_feature_dim, embed_size)
      self.sequence_pos_encoding = utils.PositionalEncoding(embed_size, dropout)
      self.attention_pool = AttentionPooling(embed_size, num_latents, dim_latent, heads, forward_expansion, dropout, device)
      self.layers = nn.ModuleList(
        [
            TransformerBlock(
            embed_size,
            heads,
            dropout = dropout,
            forward_expansion = forward_expansion,
            )  
            for _ in range(num_layers) ]
        )
      self.dropout = nn.Dropout(dropout)

   def forward(self, x, mask):
       
       motion_embedding = self.motion_embedding(x)
       out = self.sequence_pos_encoding(motion_embedding)

       for i,layer in enumerate(self.layers):
          out = layer(out, out, out, mask)
       out = self.attention_pool(out, mask)
       return out

class Text_Encoder(nn.Module):   

   def __init__(
         self,
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
      super(Text_Encoder, self).__init__()
      self.embed_size = embed_size
      self.device = device
      self.sequence_pos_encoding = utils.PositionalEncoding(embed_size, dropout)
      self.attention_pool = AttentionPooling(embed_size, num_latents, dim_latent, heads, forward_expansion, dropout, device)
      self.projection = nn.Sequential(nn.ReLU(),nn.Linear(encoded_dim, embed_size))
      self.layers = nn.ModuleList(
        [
            TransformerBlock(
            embed_size,
            heads,
            dropout = dropout,
            forward_expansion = forward_expansion,
            )  
            for _ in range(num_layers) ]
        )
      self.dropout = nn.Dropout(dropout)
   
   def forward(self, x, mask):
       x = self.projection(x)
       out = self.sequence_pos_encoding(x)
       for layer in self.layers:
          out = layer(out, out, out, mask)
       out = self.attention_pool(out, mask)
       return out

class Motion_Decoder(nn.Module):
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
         max_length
    ):
      super(Motion_Decoder, self).__init__()
      self.embed_size = embed_size
      self.device = device
      self.motion_feature_dim = motion_feature_dim
      self.embed_size = embed_size
      self.max_length = max_length
      
      self.sequence_pos_encoding = utils.PositionalEncoding(embed_size, dropout)
      self.init_pool = AttentionPooling(dim_latent, max_length, embed_size, 1, forward_expansion, dropout, device)
      self.init_projection_frames = nn.Linear(num_latents, max_length)
      self.init_projection_features = nn.Linear(dim_latent, embed_size)
      self.final_projection = nn.Linear(embed_size, motion_feature_dim)
      self.layers = nn.ModuleList(
        [
            DecoderBlock(
            embed_size,
            heads,
            forward_expansion,
            dropout,
            device
            )  
            for _ in range(num_layers) ]
        )
      self.dropout = nn.Dropout(dropout)

   def forward(self, x, z, src_mask, trg_mask):
      x = self.sequence_pos_encoding(x)
      z = z.permute(0,2,1) 
      z = self.init_projection_frames(z)
      z = z.permute(0,2,1)
      z = self.init_projection_features(z)
      for layer in self.layers:
         out = layer(x, z, z, src_mask, trg_mask)
      out = self.final_projection(out)   
      return out
   
class Motion_Encoder_tokens(nn.Module):

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
         max_length
    ):
      super(Motion_Encoder_tokens, self).__init__()
      self.embed_size = embed_size
      self.num_latents = num_latents
      self.device = device
      self.motion_feature_dim = motion_feature_dim
      self.embed_size = embed_size
      self.max_length = max_length
      self.motion_embedding = nn.Linear(motion_feature_dim, embed_size)
      self.sequence_pos_encoding = utils.PositionalEncoding(embed_size, dropout)
      self.layers = nn.ModuleList(
        [
            TransformerBlock(
            embed_size,
            heads,
            dropout = dropout,
            forward_expansion = forward_expansion,
            )  
            for _ in range(num_layers) ]
        )
      self.dropout = nn.Dropout(dropout)

      self.mu_token = nn.Parameter(torch.randn(num_latents,embed_size))
      self.logvar_token = nn.Parameter(torch.randn(num_latents,embed_size))
      self.latent_projector = nn.Linear(embed_size, dim_latent)

   def forward(self, x, mask):
       bs, nframes, nfeats = x.shape
       mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, self.mu_token.shape[0],self.mu_token.shape[1])
       logvar_token =  torch.tile(self.logvar_token, (bs,)).reshape(bs, self.logvar_token.shape[0],self.logvar_token.shape[1])
       motion_embedding = self.motion_embedding(x)

       x = torch.cat((mu_token, logvar_token, motion_embedding), 1)
   
       token_mask = torch.ones((bs, 2*self.num_latents), dtype=bool, device=x.device).unsqueeze(1).unsqueeze(2)
       mask = torch.cat((token_mask, mask), 3)
  
       out = self.sequence_pos_encoding(x)
       for i,layer in enumerate(self.layers):
          out = layer(out, out, out, mask)
       mu, logvar = out[:,0:self.num_latents,:], out[:,self.num_latents:2*self.num_latents,:]

       mu = self.latent_projector(mu)
       logvar = self.latent_projector(logvar)
       return mu, logvar

class Text_Encoder_tokens(nn.Module):   

   def __init__(
         self,
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
      super(Text_Encoder_tokens, self).__init__()
      self.embed_size = embed_size
      self.num_latents = num_latents
      self.device = device
      self.sequence_pos_encoding = utils.PositionalEncoding(embed_size, dropout)
      self.projection = nn.Sequential(nn.ReLU(),nn.Linear(encoded_dim, embed_size))
      self.layers = nn.ModuleList(
        [
            TransformerBlock(
            embed_size,
            heads,
            dropout = dropout,
            forward_expansion = forward_expansion,
            )  
            for _ in range(num_layers) ]
        )
      self.dropout = nn.Dropout(dropout)

      self.mu_token = nn.Parameter(torch.randn(num_latents,embed_size))
      self.logvar_token = nn.Parameter(torch.randn(num_latents,embed_size))
      self.latent_projector = nn.Linear(embed_size, dim_latent)

   def forward(self, x, mask):
       bs, nframes, nfeats = x.shape
       mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, self.mu_token.shape[0],self.mu_token.shape[1])
       logvar_token =  torch.tile(self.logvar_token, (bs,)).reshape(bs, self.logvar_token.shape[0],self.logvar_token.shape[1])
       text_encoding = self.projection(x)

       x = torch.cat((mu_token, logvar_token, text_encoding), 1)
       token_mask = torch.ones((bs, 2*self.num_latents), dtype=bool, device=x.device)
       mask = torch.cat((token_mask, mask), 1).unsqueeze(1).unsqueeze(2) 

       out = self.sequence_pos_encoding(x)
       for layer in self.layers:
          out = layer(out, out, out, mask)
       mu, logvar = out[:,0:self.num_latents,:], out[:,self.num_latents:2*self.num_latents,:]

       mu = self.latent_projector(mu)
       logvar = self.latent_projector(logvar)
       return mu, logvar

#----------------not needed for the task----------------
class Encoder(nn.Module):
   
   def __init__(
         self,
         embed_size,
         num_layers,
         heads,
         device,
         forward_expansion,
         dropout,
         max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
        [
            TransformerBlock(
            embed_size,
            heads,
            dropout =dropout,
            forward_expansion = forward_expansion,
            )  
            for _ in range(num_layers) ]
        )
        self.dropout = nn.Dropout(dropout)

   def forward(self, x, mask):
       
       N, seq_length = x.shape
       positions = torch. arange(0, seq_length).expand(N, seq_length).to(self.device)
       position_embedding = self.position_embedding(positions)
       out = self.dropout(self.word_embedding(x)+ position_embedding)

       for layer in self.layers:
          out = layer(out, out, out, mask)

       return out
    


class Decoder(nn.Module):
    def __init__(
         self,
         trg_vocab_size,
         embed_size,
         num_layers,
         heads,
         forward_expansion,
         dropout,
         device,
         max_length
   ):
      super(Decoder, self).__init__()
      self.decive = device
      self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
      self.position_embedding = nn.Embedding(max_length, embed_size)

      self.layers = nn.ModuleList(
        [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
         for _ in range(num_layers)]
      )
      self.fc_out = nn.Linear(embed_size, trg_vocab_size)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
       N,seq_length = x.shape
       positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
       x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

       for layer in self.layers:
         x = layer(x, enc_out, enc_out, src_mask, trg_mask)

         out = self.fc_out(x)   
         return out
    
class Transformer(nn.Module):
    def __init__(
         self,
         src_vocab_size,
         trg_vocab_size,
         src_pad_idx,
         trg_pad_idx,
         embed_size = 256,
         num_layers=6,
         forward_expansion=4,
         heads=8,
         dropout=0,
         device="cuda",
         max_length = 100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size,
                             embed_size,
                             num_layers,
                             heads,
                             device,
                             forward_expansion,
                             dropout,
                             max_length
                             )
        self.decoder = Decoder(trg_vocab_size,
                             embed_size,
                             num_layers,
                             heads,
                             forward_expansion,
                             dropout,
                             device,
                             max_length
                             )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.one((trg_len, trg_len))).expand(
           N, 1,trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
       
       src_mask = self.make_src_mask(src)
       trg_mask = self.make_trg_mask(trg)
       enc_Src = self.encoder(src,src_maks)
       out = self.decoder(trg, enc_src, src_mask, trg_mask)
       return out



if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)

   trg = torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)

   src_pad_idx = 0
   trg_pad_idx = 0
   src_vocab_size = 10
   trg_vocab_size = 10
   model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
      device
   )
   out = model(x, trg[:, :-1])
   print(out.shape)