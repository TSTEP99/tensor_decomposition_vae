"""Implementation of VAE-CP as specified in the paper https://arxiv.org/pdf/1611.00866.pdf"""
import torch
import torch.nn as nn

class VAE_CP(nn.Module):
    def __init__(self, dims, rank, K, device = None):
        """Initializes the parameters phi, theta and psi from the paper"""
       
        #Calls constructor of super class
        super(VAE_CP, self).__init__()

        #defines the device for the dataset
        if device:
            self.device = device
        else:
            self.device = "cpu"

        # Layers for computing the decoder
        self.FC_input = nn.Linear(len(dims) * rank, K)
        self.FC_mean = nn.Linear(K, 1)
        self.FC_log_var = nn.Linear(K, 1)

        #Initializes "encoder" parameters for each dimension of the tensor

        mus=[]
        lambdas=[]
        mus_tildes=[]
        lambdas_tildes=[]

        for dim in dims:
            mus.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True))) 
            lambdas.append(nn.Parameter(torch.randn((dim,rank), requires_grad=True))) 
            mus_tildes.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True)))
            lambdas_tildes.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True)))

         #parameters for the "encoder" which compute the distribution of the latent factor
        self.mus = nn.ParameterList(mus)
        self.lambdas = nn.ParameterList(lambdas)

        #parameters used to compute the prior
        self.mus_tildes = nn.ParameterList(mus_tildes)
        self.lambdas_tildes = nn.ParameterList(lambdas_tildes)

        #Activation function for the hidden layer (Tanh in the original paper) and other outputs such as element mean/sigmas
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        """Computes the forward pass using the indices of a tensor element as input"""

        #Gets the numbers of dimensions in the tensor
        num_dims = x.shape[1] 

        #First step in calculating the u vector, gets vectors that need to be concatenated
        Us = []

        #Samples u vector for each corresponding component of each dimension so that is can be concatenated
        for i in range(num_dims):
            epsilons = torch.randn((x.shape[0], self.mus[i].shape[1]), device = self.device)
            U = self.mus[i][x[:,i]] + epsilons * (torch.exp(self.lambdas[i][x[:,i]]))**(0.5) 
            Us.append(U)
        
        #Concatenates tensors to form u vector from paper 
        #More information about this step and  the previous two steps  can be found in the paper linked above
        Us = torch.concat(Us, dim=1)

        #Pass u vector through the decoder to generate mean and log_var value for each tensor element
        #Note: The original paper uses the tanh function for the hidden layer
        hidden = self.tanh(self.FC_input(Us))


        #NOTE: use ReLU Activation
        mean = self.FC_mean(hidden)

        log_var = self.FC_log_var(hidden)

        return mean, log_var

if __name__ == "__main__":

    test_tensor = torch.tensor([[1,2,3],[4,5,6],[17051,18,44]])

    dims = [17052, 19, 45]
    rank = 3
    K = 100

    model = VAE_CP(dims, rank, K)

    result = model(test_tensor)

    print(result)
