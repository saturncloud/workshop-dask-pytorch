<img style="float: right" src="img/saturn_logo.png" width="300" />

# Introduction to PyTorch with Dask

## Hands-on workshop: learn to apply Dask to improve PyTorch performance

In this hands-on workshop, attendees will have the opportunity to see how common deep learning tasks in PyTorch can be easily parallelized using Dask clusters on Saturn Cloud.

After this workshop you will know:
- Basics of how Dask works
- How to run inference with a pretrained model on Dask cluster
- How to run transfer learning on Dask cluster

To get the full learning value from this workshop, attendees should have prior experience with PyTorch. Experience with parallel computing is not needed.

## Getting started

### Steps

1. Create an account on [Saturn Cloud Hosted](https://accounts.community.saturnenterprise.io/register) or use your organization's existing Saturn Cloud Enterprise installation. 
1. Create a new project (keep defaults unless specified here)
    - Name: "workshop-dask-pytorch"
    - Image: `saturncloud/saturn-dask-pytorch:2020.11.30.16.00`
    - Workspace Settings
        - Size: `T4-4XLarge - 16 cores - 64 GB RAM - 1 GPU`
    - Click "Create"
1. Attach a Dask Cluster to the project
    - Worker Size: `T4-8XLarge - 32 cores - 128 GB RAM - 1 GPU`
    - Number of workers (n_workers): 3
    - Number of worker threads (nthreads): 3 (always the same as the number of workers when using GPU)
    - Click "Create"
1. Start both the Jupyter Server and Dask Cluster
1. Open Jupyter Lab
1. From Jupyter Lab, open a new Terminal window and clone the workshop-scaling-ml repository:
    ```bash
    git clone https://github.com/saturncloud/workshop-dask-pytorch.git /tmp/workshop-dask-pytorch
    cp -r /tmp/workshop-dask-pytorch /home/jovyan/project
    ```
1. Navigate to the "workshop-dask-pytorch" folder in the File browser and start from the [01-getting-started.ipynb](01-getting-started.ipynb) notebook.


<!-- ### Screenshots

The project from the Saturn UI should look something like this:

![project](img/project.png)

JupyterLab should look like this:

![jupyterlab](img/jupyterlab.png) -->
