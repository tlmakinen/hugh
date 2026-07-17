import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.keys = list(self.h5_file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = torch.tensor(self.h5_file[key]['data'][()], dtype=torch.float32)
        label = torch.tensor(self.h5_file[key]['label'][()], dtype=torch.long)

        sample = {'data': data, 'label': label}
        return sample

    def close(self):
        self.h5_file.close()

# Example usage
h5_file_path = 'your_dataset.h5'
batch_size = 32

# Create an instance of the custom dataset
my_dataset = H5Dataset(h5_file_path)

# Create a PyTorch DataLoader
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches
for batch in my_dataloader:
    inputs = batch['data']
    targets = batch['label']

    # Your training/validation/testing logic here
    # ...

# Don't forget to close the HDF5 file when you're done
my_dataset.close()
