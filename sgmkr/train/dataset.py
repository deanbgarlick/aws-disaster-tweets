import torch


class DisasterTweetsDataset(torch.utils.data.Dataset):
    """ turn our labels and encodings into a Dataset object. In PyTorch, this is done by subclassing a
        torch.utils.data.Dataset object and implementing __len__ and __getitem__. """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
