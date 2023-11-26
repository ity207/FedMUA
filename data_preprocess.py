# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

"""Function: load data"""
def data_init(FL_params):
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name,FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # Build a test data loader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # Evenly distribute the data into N-client copies according to the training trainset, and save all the segmented datasets in a list
    split_index = [int(trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)
    
    client_loaders = []

    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))

        '''
        By now, we have separated the local data of the client user and stored it in client_loaders.
        Each corresponds to a user's private data
        '''
    
    return client_loaders, test_loader


def data_init_non_iid(FL_params):
    import matplotlib.pyplot as plt
    kwargs = {'num_workers': 0, 'pin_memory': True} if "cuda" in FL_params.device else {}
    if FL_params.data_name == 'cifar10':
        trainset, testset = data_set(FL_params.data_name,FL_params.transform_index)
    else:
        trainset, testset = data_set(FL_params.data_name)

    # Build a test data loader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    # non-iid split
    classes = trainset.classes
    n_classes = len(classes)
    n_clients=FL_params.N_total_client

    labels = np.concatenate(
        [np.array(trainset.targets), np.array(testset.targets)], axis=0)


    client_sample_nums = balance_split(n_clients, len(trainset))
    client_idcs_dict = client_inner_dirichlet_partition(trainset.targets, n_clients, n_classes, dir_alpha=1, client_sample_nums=client_sample_nums, verbose=False)
    
    client_idcs = []
    for _ in client_idcs_dict.values():
        client_idcs.append(_)


    client_dataset = [torch.utils.data.Subset(trainset, indices) for indices in client_idcs]

    # This section describes how different labels are assigned to different clients
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_clients + 1.5, 1),
                label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                        c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(f"{FL_params.data_name}_noniid_plot.png")

    client_loaders = []

    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
    
        '''
        By now, we have separated the local data of the client user and stored it in client_loaders.
        Each corresponds to a user's private data
        '''
    
    return client_loaders, test_loader


def balance_split(num_clients, num_samples):
    """Assign same sample sample for each client.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    return client_sample_nums


def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha,
                                     client_sample_nums, verbose=True):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                       size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict


def data_set(data_name,transform_index=False):
    if not data_name in ['mnist','purchase','adult','cifar10']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')
    
    #model: 2 conv. layers followed by 2 FC layers
    if(data_name == 'mnist'):
        trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        testset = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        
    #model: Similar to vgg network
    elif(data_name == 'cifar10'):

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # converting images to tensor
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
        # if the image dataset is black and white image, there can be just one number. 
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])

 
        trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)


        testset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    
    #model: 2 FC layers
    elif(data_name == 'purchase'):
        xx = np.load("./data/purchase/purchase_xx.npy")
        yy = np.load("./data/purchase/purchase_y2.npy")
        # yy = yy.reshape(-1,1)
        # enc = preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(yy)
        # yy = enc.transform(yy).toarray()
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
        
        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)
        
        trainset = TensorDataset(X_train_tensor,y_train_tensor)
        testset = TensorDataset(X_test_tensor,y_test_tensor)
        
        trainset.classes = ["0","1"]
        testset.classes = ["0","1"]

        trainset.targets = y_train_tensor
        testset.targets = y_test_tensor

        
    return trainset, testset



    


