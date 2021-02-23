from dask_saturn import SaturnCluster
from dask.distributed import Client
import matplotlib.pyplot as plt
import numpy as np
import os

import math
import datetime
import json 
import pickle
import tensorboard
#tensorboard.__version__
from dask_pytorch_ddp import data, dispatch
from torch.utils.data.sampler import SubsetRandomSampler

import s3fs
import re

cluster = SaturnCluster()
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
    whole_dataset = data.S3ImageFolder(bucket, prefix, transform=transform, anon = True)
    return whole_dataset
            
            
def get_splits_parallel(train_pct, data, batch_size, subset = False, workers = 1):
    '''Select two samples of data for training and evaluation'''
    classes = data.classes
    train_size = math.floor(len(data) * train_pct)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:len(data)]

    if subset:
        train_idx = np.random.choice(train_idx, size = int(np.floor(len(train_idx)*(1/workers))), replace=False)
        test_idx = np.random.choice(test_idx, size = int(np.floor(len(test_idx)*(1/workers))), replace=False)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork'))
    test_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork'))
    
    return train_loader, test_loader


def replace_label(dataset_label, model_labels):
    label_string = re.search('n[0-9]+-([^/]+)', dataset_label).group(1)
    
    for i in model_labels:
        i = str(i).replace('{', '').replace('}', '')
        model_label_str = re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(i))
        model_label_idx = re.search('''b["']([0-9]+):''', str(i)).group(1)
        
        if re.search(str(label_string).replace('_', ' '), str(model_label_str).replace('_', ' ')):
            return i, model_label_idx
            break


#####


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
        
## Text parsing

def format_labels(label, pred):
    pred = str(pred).replace('{', '').replace('}', '')

    if re.search('n[0-9]+-([^/]+)', str(label)) is None:
        label = re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(label)).group(1)
    else: 
        label = re.search('n[0-9]+-([^/]+)', str(label)).group(1)
    
    if re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(pred)) is None:
        pred = re.search('n[0-9]+-([^/]+)', str(pred)).group(1)
    else:
        pred = re.search('''b["'][0-9]+: ["']([^\/]+)["'],["']''', str(pred)).group(1)
    return(label, pred)

def is_match(label, pred):
    ''' Evaluates human readable prediction against ground truth.'''
    if re.search(str(label).replace('_', ' '), str(pred).replace('_', ' ')):
        match = True
    else:
        match = False
    return(match)

## Pred Parsing

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    batch = net(images)
    _, preds_tensor = torch.max(batch, 1)
    preds = preds_tensor.cpu().numpy()
    perct = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, batch)]

    return preds, perct

def plot_classes_preds(net, images, labels, preds_tensors, perct, trainclasses):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds = preds_tensors.cpu().numpy()
    pred_class_set = [trainclasses[i] for i in preds]
    lab_class_set = [trainclasses[i] for i in labels]
    
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 24))
    plt.subplots_adjust(wspace = 0.6)
    
    for idx in np.arange(4):   
        raw_label = lab_class_set[idx]
        raw_pred = pred_class_set[idx]

        label, pred = format_labels(raw_label,raw_pred)
        
        ax = fig.add_subplot(2, 2, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            pred, perct[idx]*100, label), color=("green" if is_match(label, pred) else "red"))

    return fig

            
from IPython.display import display, HTML
gpu_links = f'''
<b>Cluster Dashboard links</b>
<ul>
<li><a href="{client.dashboard_link}/status" target="_blank">CPU dashboard</a></li>
<li><a href="{client.dashboard_link}/individual-gpu-utilization" target="_blank">GPU utilization</a></li>
<li><a href="{client.dashboard_link}/individual-gpu-memory" target="_blank">GPU memory</a></li>
</ul>
'''