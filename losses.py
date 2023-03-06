"""File containing the loss functions used in training"""
import torch

def reconstruction_loss(elements, means, log_vars):
    """Computes the reconstruction loss specified in VAE-CP paper"""
    L = means.shape[0]
    std = torch.sqrt(torch.exp(log_vars))
    return (-torch.log(std) - 0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * ((elements - means)/std) ** 2 ).mean()

def regularization_loss(mus, lambdas, mus_tildes, lambdas_tildes):
    """Computes the regularization loss specified in VAE-CP paper"""
    loss = 0
    for i in range(len(mus)):
        mu = mus[i]
        lambda_ = lambdas[i]
        mu_tilde = mus_tildes[i]
        lambda_tilde = lambdas_tildes[i]

        var = torch.exp(lambda_)
        var_tilde = torch.exp(lambda_tilde)
        std = torch.sqrt(var)
        std_tilde = torch.sqrt(var_tilde)
        loss += (torch.log(std_tilde/std) + (var + (mu - mu_tilde)**2)/(2 * var_tilde) - 0.5).sum()
    return loss

def compute_total_variation_loss(mu):
    """Computes the total variation term to used in the total variation loss function"""
    return torch.pow(mu[1:] - mu[:-1], 2).sum()

def compute_laplacian_loss(mu):
    pass

def original_loss(elements, means, log_vars, mus, lambdas, mus_tildes, lambdas_tildes):
    """The original loss function specified in the VAE_CP paper"""
    return -reconstruction_loss(elements, means, log_vars) + regularization_loss(mus, lambdas, mus_tildes, lambdas_tildes)

def total_variation_loss(elements, means, log_vars, mus, lambdas, mus_tildes, lambdas_tildes, dims = [2]):
    """The loss function with a total variation term added"""
    loss = original_loss(elements, means, log_vars, mus, lambdas, mus_tildes, lambdas_tildes)

    for i in range(len(mus)):
        if i in dims:
            loss += compute_total_variation_loss(mus[i])

    return loss
