import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from sklearn.model_selection import StratifiedShuffleSplit


class SimpleTensorDataset(Dataset):
    def __init__(self, data, targets, transformation=None):
        """
        Args:
            data (Tensor): A tensor containing the data e.g. images
            targets (Tensor): A tensor containing all the labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transformation = transformation

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        if self.transformation:
            img = self.transformation(img)

        return img, label

    def __len__(self):
        return len(self.data)


def load_to_memory(data_loader, data_len):
    """
    Return full-batch tensors containing all the data in data_loader
    which of length data_len
    """

    # Preallocate memory for data
    X_data, y_data = torch.Tensor(
        data_len, 3, 64, 64), torch.LongTensor(data_len)

    # Load data into memory
    idx = 0
    for X_, y_ in data_loader:
        bsize = X_.size(0)
        X_data[idx:idx+bsize, ...] = X_
        y_data[idx:idx+bsize] = y_
        idx += bsize

    return X_data, y_data


def make_dataloaders(train_dataset, test_dataset, train_transform, validation_transform, batch_size, test_batch_size, val_frac=0.1, num_workers=0, seed=None):
    """
    Create dataloaders for input training and test datasets. Training data is preloaded into 
    memory for faster access during training

    train_dataset must have only had torchvision.transforms.ToTensor() transform applied
    """

    # create a random split for the training dataset
    shuffler = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed).split(
        np.zeros(len(train_dataset)), train_dataset.targets)

    indices = [(train_idx, validation_idx)
               for train_idx, validation_idx in shuffler][0]

    # Random samplers for the training and validation dataloaders
    sampler_train = SubsetRandomSampler(indices[0])
    sampler_val = SubsetRandomSampler(indices[1])

    # Instantiate dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=sampler_train,
                              shuffle=False,
                              num_workers=num_workers)

    validation_loader = DataLoader(train_dataset,
                                   batch_size=test_batch_size,
                                   sampler=sampler_val,
                                   shuffle=False,
                                   num_workers=num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    # Calculate length of tensor required to store training data
    train_data_len = len(train_dataset)
    val_data_len = int(train_data_len*val_frac)
    train_data_len = train_data_len - val_data_len

    # Create tensors containing all the training and validation data
    X_train, y_train = load_to_memory(train_loader, train_data_len)
    X_val, y_val = load_to_memory(validation_loader, val_data_len)

    # Instantiate new datasets
    train_dataset_ = SimpleTensorDataset(X_train, y_train,
                                         transformation=train_transform)
    validation_dataset_ = SimpleTensorDataset(X_val, y_val,
                                              transformation=validation_transform)

    # Overwrite training and validation with new dataloaders
    train_loader = DataLoader(train_dataset_,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    validation_loader = DataLoader(validation_dataset_,
                                   batch_size=test_batch_size,
                                   shuffle=False,
                                   num_workers=num_workers)

    return train_loader, validation_loader, test_loader
