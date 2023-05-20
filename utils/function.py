import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
# import torchvision
from torchvision import transforms
import copy
import numpy as np
import os
import json
import pickle
from PIL import Image

import matplotlib.pyplot as plt

##########

def clip(v):
    v_norm = torch.norm(v)
    scale = min(1, 100 / v_norm)
    return v * scale

def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def smoothed_weiszfeld(weights, alphas):
    m = len(weights)
    nu = 0.1
    T = 8
    z = torch.zeros_like(weights[0])

    if alphas.numel() != m:
        raise ValueError

    if nu < 0:
        raise ValueError
    
    for t in range(T):
        betas = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, weights[k])
            betas.append(alphas[k] / max(distance, nu))
        
        z = torch.zeros_like(weights[0])
        for w, beta in zip(weights, betas):
            z += w * beta
        z /= sum(betas)
    return z
            
def bucketing_wrapper(inputs, num_samples, s, M):
    """
    Key functionality.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # print("Using bucketing wrapper.")
    indices = list(range(len(inputs)))
    np.random.shuffle(indices)

    T = int(np.ceil(M / s))

    reshuffled_inputs = []
    reshuffled_samples = []
    with torch.no_grad():
        for t in range(T):
            indices_slice = indices[t * s : (t + 1) * s]
            g_bar = copy.deepcopy(inputs[0].m).to(device)
            g_bar_sample = num_samples[indices_slice].sum()
            for name, params in g_bar.named_parameters():
                for i in indices_slice:
                    inputs[i].m.to(device)
                    params.data += inputs[i].m.state_dict()[name].data * num_samples[i]
                params /= num_samples[indices_slice].sum()

            reshuffled_inputs.append(g_bar)
            reshuffled_samples.append(g_bar_sample)
            
    return reshuffled_inputs, torch.tensor(reshuffled_samples,device=device)

############### observation ##########
def observe_record(X, x_avg,lr, round, threshold_l, norm_l):
    if round in threshold_l:
        temp_norm_l = []
        for w in X:
            temp_norm_l.append(grad_norm(w.x,x_avg,lr))
        norm_l.append(temp_norm_l)
    return norm_l

def grad_norm(x, x_avg, lr, norm_type=2):
    with torch.no_grad():
        total_norm= 0.0
        # No need to include the learning rate as it is a constant within
        # the local computation rounds.
        for name, params in x_avg.named_parameters():
            param_norm = torch.norm(((params.data - x.state_dict()[name].data)).detach(),norm_type)
            total_norm += param_norm.item() ** 2 
    return total_norm ** 0.5 / lr

############## dropout ##################
def counting(num_samples, active_client0, active_client1):
    return [num_samples[active_client0].tolist(), num_samples[active_client1].tolist()]

############## post processing #########
def summary_save(algorithm,tb, num_sample_list, dataset, alpha, beta0, iid, seed,bs,lr,global_round,local_round, K,epsilon, Beta, balanced, seed1, seed2, **kwargs):
    cur_path = os.path.abspath(os.curdir)

    summary_path0 = os.path.join(cur_path,'results/{}'.format(dataset),'data_{}_{}_{}'.format(alpha,beta0,iid),'data')
    os.makedirs(summary_path0, exist_ok=True)
    summary_path = os.path.join(summary_path0,\
                            'alg_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(algorithm,seed,seed1,seed2,bs,lr,global_round,local_round, K,epsilon, Beta, balanced))

    with open(summary_path,'w') as f:
        json.dump(tb, f)
    
    summary_path01 = os.path.join(cur_path,'results/{}'.format(dataset),'data_{}_{}_{}'.format(alpha,beta0,iid),'sample')
    
    os.makedirs(summary_path01, exist_ok=True)

    summary_path1 = os.path.join(summary_path01,\
                            'sample_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(algorithm,seed,seed1,seed2,bs,lr,global_round,local_round, K,epsilon, Beta, balanced))

    with open(summary_path1,'w') as f1:
        json.dump(num_sample_list, f1)

# define a summary save function
def observe_save(algorithm, num_sample_list, norm_l,dataset, alpha, beta0, iid,seed,bs,lr,global_round,local_round, K, M,epsilon, Beta, balanced, seed1, seed2, **kwargs):
    cur_path = os.path.abspath(os.curdir)
    summary_path0 = os.path.join(cur_path,'results/{}'.format(dataset),'data_{}_{}_{}'.format(alpha,beta0,iid),'observations')
    os.makedirs(summary_path0, exist_ok=True)
    summary_path = os.path.join(summary_path0,\
                            'obs_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_M_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(algorithm,seed,seed1,seed2,bs,lr,global_round,local_round, K, M, epsilon, Beta, balanced))

    with open(summary_path,'w') as f:
        json.dump([norm_l, num_sample_list[0]], f)

########### load dataset ##########
# define an auxillary function to load the dataset into PyTorch.
# torch.TensorDataset = load_dataset(x:list,y:list)
def load_dataset(x,y):
    tensor_x = torch.Tensor(np.array(x)) # transform to torch tensor
    tensor_y = torch.Tensor(np.array(y))
    t_set = TensorDataset(tensor_x,tensor_y) # create your datset
    return t_set

# define a loader function to load Dataset into DataLoader
# DataLoader = loader(torch.TensorDataset, int)
def loader(dataset, batch_size):
    return DataLoader(dataset, batch_size = batch_size, drop_last=True)

class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target

def load_local_data(seed1,seed2,alpha,beta0,iid,bs,balanced, M, dataset,**kwargs):
    if dataset == 'synthetic':
        # define the paths (train and test data)
        cur_path = os.path.abspath(os.curdir)
        train_path = os.path.join(cur_path,'dataset/synthetic','data_{}_{}_{}_{}_{}'.format(alpha,beta0,iid, seed1, seed2),'train/mytrain.json')
        test_path = os.path.join(cur_path,'dataset/synthetic','data_{}_{}_{}_{}_{}'.format(alpha,beta0,iid, seed1, seed2),'test/mytest.json')

        # Open the train JSON file
        with open(train_path) as f:
            # Load the data from the file
            train_data = json.load(f)

        # Open the test JSON file
        with open(test_path) as f:
            # Load the data from the file
            test_data = json.load(f)

        # reconstruct the train dataset from local files:
        # tensor_train_dataset_split: list of the items being individual tensor datasets;
        tensor_train_dataset_split = []
        train_dataset_x_list, train_dataset_y_list, train_loader_split  = [], [], [] 

        # num_samples: list of the items being the number of batches;
        if torch.cuda.is_available():
            num_samples = torch.tensor(train_data['num_samples'],device='cuda')
        else:
            num_samples = torch.tensor(train_data['num_samples'])
            
        if balanced == 0 and dataset == 'synthetic':
            if torch.cuda.is_available():
                train_bs = copy.deepcopy(num_samples).cuda()
            else:
                train_bs = copy.deepcopy(num_samples)
        elif balanced == 0 and dataset == 'cifar10':
            if torch.cuda.is_available():
                train_bs = copy.deepcopy(num_samples/10).type(int).cuda()
            else:
                train_bs = copy.deepcopy(num_samples/10).type(int)
        else:
            if torch.cuda.is_available():
                train_bs = torch.tensor([bs for i in range(M)], device='cuda')
            else:
                train_bs = torch.tensor([bs for i in range(M)])
                
        count0 = 0
        for _ , item in train_data['user_data'].items():
            tensor_train_dataset_split.append(load_dataset(item['x'], item['y']))
            train_loader_split.append(loader(tensor_train_dataset_split[-1],int(train_bs[count0])))
            train_dataset_x_list.extend(item['x'])
            train_dataset_y_list.extend(item['y'])
            count0 += 1

        # tensor_train_dataset: one tensor dataset;
        # train_loader: one DataLoader
        tensor_train_dataset = load_dataset(train_dataset_x_list,train_dataset_y_list)
        train_loader = loader(tensor_train_dataset, bs)

        # tensor_test_dataset: one tensor dataset;
        # test_loader: one DataLoader
        tensor_test_dataset = load_dataset(test_data['user_data']['x'],test_data['user_data']['y'])
        test_loader = loader(tensor_test_dataset, bs)

    elif dataset == 'cifar10':
        # define the paths (train and test data)
        cur_path = os.path.abspath(os.curdir)
        train_path = os.path.join(cur_path,'dataset/cifar10','dirichlet_{}_{}'.format(alpha,seed1),'data/train','all.pkl')
        test_path = os.path.join(cur_path,'dataset/cifar10','dirichlet_{}'.format(alpha,seed1),'data/test','all.pkl')

        # Open the train JSON file
        with open(train_path,'rb') as f:
            # Load the data from the file
            train_data = pickle.load(f)

        # Open the test JSON file
        with open(test_path,'rb') as f:
            # Load the data from the file
            test_data = pickle.load(f)

        # reconstruct the train dataset from local files:
        # tensor_train_dataset_split: list of the items being individual tensor datasets;
        tensor_train_dataset_split = []
        train_dataset_x_list, train_dataset_y_list, train_loader_split  = [], [], [] 

        # num_samples: list of the items being the number of batches;
        if torch.cuda.is_available():
            num_samples = torch.tensor(train_data['num_samples'],device='cuda')
        else:
            num_samples = torch.tensor(train_data['num_samples'])
            
        if balanced == 0:
            if torch.cuda.is_available():
                train_bs = copy.deepcopy(num_samples).cuda()
            else:
                train_bs = copy.deepcopy(num_samples)
        else:
            if torch.cuda.is_available():
                train_bs = torch.tensor([bs for i in range(M)], device='cuda')
            else:
                train_bs = torch.tensor([bs for i in range(M)])
                
        count0 = 0
        for _ , item in train_data['user_data'].items():
            tensor_train_dataset_split.append(MiniDataset(item['x'], item['y']))
            train_loader_split.append(loader(tensor_train_dataset_split[-1],int(train_bs[count0])))
            train_dataset_x_list.extend(item['x'])
            train_dataset_y_list.extend(item['y'])
            count0 += 1

        # tensor_train_dataset: one tensor dataset;
        # train_loader: one DataLoader
        tensor_train_dataset = MiniDataset(train_dataset_x_list,train_dataset_y_list)
        train_loader = loader(tensor_train_dataset, bs)

        # tensor_test_dataset: one tensor dataset;
        # test_loader: one DataLoader
        tensor_test_dataset = MiniDataset(test_data['user_data']['x'],test_data['user_data']['y'])
        test_loader = loader(tensor_test_dataset, bs)

    return train_bs, train_loader, test_loader, train_loader_split

##### visualization ####
def dropout_plot(FedAvg_config, FedProx_config, seed_l, threshold):
    l_l, l1_l = [], []
    for _ in seed_l:
        a = count(['FedAvg_v'],**FedAvg_config)
        b = count(['FedProx_v'],**FedProx_config)
        base = 10 * np.array(a[0]).sum() / 100
        base1 = 10 * np.array(b[0]).sum() / 100
        l, l1 = [], []
        for i in a[1:]:
            l.append((np.array(i[0]).sum() - np.array(i[1]).sum()) / base)
        for i in b[1:]:
            l1.append((np.array(i[0]).sum() - np.array(i[1]).sum()) / base1)
        l_l.append(l[:threshold])
        l1_l.append(l[:threshold])

    l_l, l1_l = np.array(l_l), np.array(l1_l)
    _,ax = plt.subplots(1,2,figsize=(22,2))

    ax[0].fill_between(np.arange(len(l_l[0])),np.min(l_l,axis=0), np.max(l_l,axis=0), alpha=0.3)
    ax[0].plot(np.arange(len(l))[:threshold],np.mean(l_l,axis=0))
    ax[0].set_title(r'FedAvg variant dropout $\epsilon$')

    ax[1].fill_between(np.arange(len(l_l[0])),np.min(l1_l,axis=0), np.max(l1_l,axis=0), alpha=0.3)
    ax[1].plot(np.arange(len(l1))[:threshold],np.mean(l1_l,axis=0))
    ax[1].set_title(r'FedProx variant dropout $\epsilon$')

def load_summary_path(algorithms,seeds, local_round,dataset, alpha, beta0, iid, seed1, seed2,bs, \
                          lr, global_round, K, epsilon, Beta, balanced, **kwargs):
    if algorithms in ['FedAvg','FedProx_v','MIFA']:
        cur_path = os.path.abspath(os.curdir)
        summary_path0 = os.path.join(cur_path,'results',dataset,'data_{}_{}_{}'\
                                    .format(alpha,beta0,iid),'data')
        summary_path = os.path.join(summary_path0,\
                                'alg_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'\
                                        .format(algorithms,seeds,seed1,seed2, bs,lr,global_round,\
                                                local_round, K,epsilon, 10.0, balanced))
    else:
        cur_path = os.path.abspath(os.curdir)
        summary_path0 = os.path.join(cur_path,'results',dataset,'data_{}_{}_{}'\
                                    .format(alpha,beta0,iid),'data')
        summary_path = os.path.join(summary_path0,\
                                'alg_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'\
                                        .format(algorithms,seeds,seed1,seed2, bs,lr,global_round,\
                                                local_round, K,epsilon, Beta, balanced))
    return summary_path

def results_visualization_l(alg,configs,length, seed_l,smoothed):
    fig,ax = plt.subplots(1,2,figsize=(22,6))
    for algorithm, config in zip(alg,configs):
        train_l, test_a = [],[]
        for seeds in seed_l:
            summary_path = load_summary_path(algorithm,seeds,**config)
            with open(summary_path) as f:
                tb = json.load(f) 
            epoch = tb['epoch']
            if smoothed:
                train_l.append(np.array(tb['train loss'][:length])[np.arange(0,length,smoothed, dtype=int).astype(int)])
                test_a.append(np.array(tb['test acc'][:length])[np.arange(0,length,smoothed, dtype=int).astype(int)])
            else:
                train_l.append(tb['train loss'][:length])
                test_a.append(tb['test acc'][:length])
        
        if smoothed:
            epoch = np.array(epoch[:length])[np.arange(0,length,smoothed, dtype=int).astype(int)]
        else:
            epoch = epoch[:length]

        train_data = np.array(train_l)
        test_data = np.array(test_a)
        
        train_mean = np.mean(train_data,axis=0)
        test_mean = np.mean(test_data,axis=0)
        handles1, labels1 = [], []

        if algorithm not in ['FedAvg_v','FedProx_v']:
            if algorithm == 'gm':
                algorithm = 'GM'
            elif algorithm == 'bucketing_gm':
                algorithm = 'bucketing_GM'
            ax[0].fill_between(epoch,np.min(train_data,axis=0)[:length], np.max(train_data,axis=0)[:length], alpha=0.3)
            ax[0].plot(epoch,train_mean,label=algorithm,linestyle='--')
            ax[1].fill_between(epoch,np.min(test_data,axis=0)[:length], np.max(test_data,axis=0)[:length], alpha=0.3)
            ax[1].plot(epoch,test_mean,label=algorithm,linestyle='--')
            h1, l1 = ax[1].get_legend_handles_labels()
            handles1 += h1
            labels1 += l1
        else:
            if algorithm == 'FedProx_v':
                algorithm = 'FedProx variant'
            elif algorithm == 'FedAvg_v':
                algorithm = 'FedAvg variant'
            ax[0].fill_between(epoch,np.min(train_data,axis=0)[:length], np.max(train_data,axis=0)[:length], alpha=0.3)
            ax[0].plot(epoch,train_mean[:length],label=algorithm)
            ax[1].fill_between(epoch,np.min(test_data,axis=0)[:length], np.max(test_data,axis=0)[:length], alpha=0.3)
            ax[1].plot(epoch,test_mean[:length],label=algorithm)#+' {}'.format(Beta)
            h1, l1 = ax[1].get_legend_handles_labels()
            handles1 += h1
            labels1 += l1
    # set titles, legends
    ax[0].set_ylabel('train loss')
    ax[1].set_ylabel('test accuracy')
    fig.legend(handles1, labels1, loc='upper center',ncol=len(alg), bbox_to_anchor=(0.5, 1.1), frameon=True)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()

#### dropout mass count
def count(alg,dataset,alpha,beta0,iid, epsilon,seed,seed1,seed2,bs,lr,global_round,local_round,K,balanced,Beta,**kwargs):
    for algorithm in alg:#'synthetic'
        if algorithm in ['FedAvg','FedProx','MIFA']:
            cur_path = os.path.abspath(os.curdir)
            summary_path0 = os.path.join(cur_path,'results',dataset,'data_{}_{}_{}'\
                                        .format(alpha,beta0,iid),'sample')
            summary_path = os.path.join(summary_path0,\
                                    'sample_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'\
                                            .format(algorithm,seed,seed1,seed2, bs,lr,global_round,\
                                                    local_round, K,epsilon, 10.0, balanced))
        else:
            cur_path = os.path.abspath(os.curdir)
            summary_path0 = os.path.join(cur_path,'results',dataset,'data_{}_{}_{}'\
                                        .format(alpha,beta0,iid),'sample')
            summary_path = os.path.join(summary_path0,\
                                    'sample_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'\
                                            .format(algorithm,seed,seed1,seed2, bs,lr,global_round,\
                                                    local_round, K,epsilon, Beta, balanced))

        with open(summary_path) as f:
            tb = json.load(f) 
        return tb