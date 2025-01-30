import torch
import torch.nn.functional as F
import torch.distributions as dist

def reconstruction_loss_smooth_l1(predicted_motion, ground_truth_motion):
    """
    Computes the reconstruction loss using Smooth L1 Loss (Huber Loss).
    
    Args:
        predicted_motion (torch.Tensor): Predicted motion tensor of shape (batch_size, seq_len, num_features).
        ground_truth_motion (torch.Tensor): Ground truth motion tensor of the same shape.

    Returns:
        torch.Tensor: Reconstruction loss (scalar).
    """
    return F.smooth_l1_loss(predicted_motion, ground_truth_motion)

def kl_divergence_loss(mu1, logvar1, mu2, logvar2):
    """
    Computes the KL divergence between two diagonal Gaussian distributions.
    
    Args:
        mu1 (torch.Tensor): Mean of the first distribution (batch_size, latent_dim).
        logvar1 (torch.Tensor): Log-variance of the first distribution, same shape as mu1.
        mu2 (torch.Tensor): Mean of the second distribution, same shape as mu1.
        logvar2 (torch.Tensor): Log-variance of the second distribution, same shape as mu1.

    Returns:
        torch.Tensor: KL divergence loss (scalar).
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    dist1 = dist.Normal(mu1, var1.sqrt())
    dist2 = dist.Normal(mu2, var2.sqrt())
    
    kl_loss = dist.kl.kl_divergence(dist1, dist2)  # Element-wise KL
    return kl_loss.mean() 

def mixed_latent_loss(mu_text, logvar_text, mu_motion, logvar_motion):
   
    kl_motion_text = kl_divergence_loss(mu_motion, logvar_motion, mu_text, logvar_text)
    kl_text_motion = kl_divergence_loss(mu_text, logvar_text, mu_motion, logvar_motion )

    mu_prior = torch.zeros_like(mu_text)
    var_prior = torch.ones_like(logvar_text) 
    kl_text_isotropic = kl_divergence_loss(mu_text, logvar_text, mu_prior, var_prior)
    kl_motion_isotropic = kl_divergence_loss(mu_motion, logvar_motion, mu_prior, var_prior)

    return kl_text_isotropic + kl_motion_isotropic + kl_motion_text + kl_text_motion

def cross_modal_loss(z_text, z_motion):
    return F.smooth_l1_loss(z_text, z_motion)

def train_loss(motion_from_text, motion_from_motion, ground_truth_motion, z_text, z_motion, mu_text, var_text, mu_motion, var_motion):
    L1 = reconstruction_loss_smooth_l1(motion_from_text, ground_truth_motion)
    L2 = reconstruction_loss_smooth_l1(motion_from_motion, ground_truth_motion)
    KL = mixed_latent_loss(mu_text, var_text, mu_motion, var_motion)
    cross = cross_modal_loss(z_text, z_motion)
    return L1 + L2 + 1e-6*KL + cross


