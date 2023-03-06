"""File used to train VAE-CP"""
from data import TensorDataset
from losses import original_loss, total_variation_loss
from math import floor
from preprocess import create_indices, process_eegs
from torchmetrics import MeanSquaredError
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from vae_cp import VAE_CP
import torch

BETA = 100

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for batch, (indices, elements) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        means, log_vars = model(indices)
        loss = loss_fn(elements, means, log_vars, model.mus, model.lambdas, model.mus_tildes, model.lambdas_tildes)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # if batch % 100 == 0:
        #     loss = loss.item()
        #     print(f"loss: {loss:>7f}")

    train_loss /= num_batches
    print(f"train loss: {train_loss:}  \n")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mean_squared_error = MeanSquaredError(squared = False).to(model.device)
    test_loss, rmse_loss = 0, 0
    model.eval()

    with torch.no_grad():
        for indices, elements in dataloader:
            means, log_vars = model(indices)
            test_loss += loss_fn(elements, means, log_vars, model.mus, model.lambdas, model.mus_tildes, model.lambdas_tildes).item()
            rmse_loss += mean_squared_error(elements, means)
    
    test_loss /= num_batches
    rmse_loss /= num_batches
    print(f"test loss: {test_loss:}  \n")
    print(f"rmse loss:{rmse_loss}  \n")

if __name__ == "__main__":

    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {DEVICE} device")

    DEVICE = "cuda:1"

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    HIDDEN_LAYER_SIZE = 100
    RANK = 3

    full_psds, _, _, _, _, grade, epi_dx, alz_dx, _, _, _, _ = process_eegs()

    pop_psds= full_psds[(epi_dx<0) & (alz_dx<0)]

    pop_psds /= (torch.max(pop_psds) - torch.min(pop_psds))

    indices = create_indices(pop_psds.shape)
    indices = indices.to(torch.long)
    indices = indices.to(DEVICE)

    dims = pop_psds.shape

    print("Dimensions of population tensor:", dims)

    flat_psds = pop_psds.reshape((-1,1))
    flat_psds = flat_psds.to(DEVICE)

    total_length = len(indices)

    train_length = floor(0.8 * total_length)
    val_length = floor( 0.5 * (total_length-train_length))
    test_length = total_length - train_length - val_length

    lengths = [train_length, val_length, test_length]

    dataset= TensorDataset(flat_psds, indices)

    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths, generator = torch.Generator().manual_seed(42))

    print(f"Training Set has length {train_dataset.__len__()}")
    print(f"Validation Set has length {val_dataset.__len__()}")
    print(f"Test Set has length {test_dataset.__len__()}")

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = VAE_CP(dims, rank = RANK, K = HIDDEN_LAYER_SIZE, device = DEVICE)

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9, 0.999), eps=1e-8)

    # Note: May need to check to make sure the model parameters are being updated
    # for parameter in model.parameters():
    #     print(parameter.shape)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, total_variation_loss, optimizer)
        test_loop(test_dataloader, model, total_variation_loss)
        torch.save(model, f'checkpoints/vae_cp_epoch_{t+1}.pth')
    print("Done!")
