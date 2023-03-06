import torch
from preprocess import create_indices, process_eegs
from torch.utils.data import Dataset
from tqdm import trange

class TensorDataset(Dataset):
    """Custom Dataset for the tensor, may need to add more functionality later on"""
    def __init__(self, elements, indices):
        self.elements = elements
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index_array = self.indices[idx]
        element = self.elements[idx]

        return index_array, element

if __name__ == "__main__":
    #Note to self: will need to migrate these tests to another file
    
    returned_arrays = process_eegs()

    full_psds = returned_arrays[0]
    grade = returned_arrays[5]
    epi_dx = returned_arrays[6]
    alz_dx = returned_arrays[7]

    psds_min = torch.min(full_psds).item()
    psds_max = torch.max(full_psds).item()

    trans_psds = full_psds/(psds_max-psds_min)

    indices = create_indices(trans_psds.shape)

    flat_psds = trans_psds.reshape((-1,1))

    test_dataset = TensorDataset(flat_psds, indices)

    length = test_dataset.__len__()

    for i in trange(length):
        test_dataset.__getitem__(i)

