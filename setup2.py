from dask_saturn import SaturnCluster
from dask.distributed import Client

cluster = SaturnCluster(
    n_workers = 3, 
    scheduler_size = 'g4dn4xlarge', 
    worker_size = 'g4dn8xlarge'
)
client = Client(cluster)
client.wait_for_workers(3)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dask_pytorch_ddp import results, data, dispatch
from torch.utils.data.sampler import SubsetRandomSampler

def prepro_batches(bucket, prefix):
    '''Initialize the custom Dataset class defined above, apply transformations.'''
    transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(250), 
    transforms.ToTensor()])
    whole_dataset = data.S3ImageFolder(bucket, prefix, transform=transform)
    return whole_dataset

def get_splits_parallel(train_pct, data, batch_size):
    '''Select two samples of data for training and evaluation'''
    classes = data.classes
    train_size = math.floor(len(data) * train_pct)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:len(data)]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork'))
    test_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork'))
    
    return train_loader, test_loader