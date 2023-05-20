from abc import ABC, abstractmethod
from backend import client, server
import numpy as np
import os
import copy
import json
from utils import observe_record, counting,load_local_data
import torch

class trainer_base(ABC):
    def __init__(self,config,observation,K, M, lr, lr_decay,algorithm, Beta, alpha, beta0, iid,\
                      balanced, epsilon, dataset, local_round, global_round, seed, seed1, seed2, \
                        bs,print_every,**kwargs):
        self.train_bs, self.train_loader, self.test_loader, self.train_loader_split = load_local_data(**config)
        self.num_sample_list = [self.train_bs.detach().cpu().tolist()]
        self.config = config
        self.print_every = print_every
        self.observation = observation
        self.algorithm = algorithm
        self.Beta = Beta
        self.K, self.M = K, M
        self.local_round = local_round
        self.global_round = global_round
        self.dataset = dataset
        self.alpha = alpha
        self.beta0 = beta0
        self.seed, self.seed1, self.seed2 = seed, seed1, seed2
        self.bs = bs
        self.epsilon = epsilon
        self.balanced = balanced
        self.iid = iid
        self.lr_latest, self.lr = lr, lr
        self.lr_decay = lr_decay
        if torch.cuda.is_available():
            self.active_client = torch.arange(self.M).cuda()
        else:
            self.active_client = torch.arange(self.M)
        if observation:
            # norm_l:  norm list
            # threshold_l: threshold list
            self.norm_l = []
            self.threshold_l = [int(item) for item in config['threshold_l'].split(',')] 
        self.seed_init()
        self.X = [client(self.train_loader_split[idx],**config)for idx in range(M)]
        self.server = server(self.train_loader,self.test_loader,self.train_bs,**config)
      
    
    def seed_init(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def sample(self):
        if torch.cuda.is_available():
            raw = torch.randperm(self.M,device='cuda')
        else:
            raw = torch.randperm(self.M)
        self.active_client = raw[:self.K]

    def dropout(self):
        num_base = int(self.epsilon * self.K * self.train_bs.sum() / self.M)

        file_path = 'index_list_{}.json'.format(self.alpha)
        assert open(file_path), 'calC does not exist, please build up one first!'
        with open(file_path) as f:
            list = json.load(f)
            
        count1 = 0
        retain = copy.deepcopy(self.active_client)

        for ind,i in enumerate(self.active_client.detach().cpu().numpy()):
            if self.train_bs[i] > num_base:
                continue
            elif (i in list) and (count1 + self.train_bs[i] < num_base) and (retain.numel() > 1):
                    retain = retain[retain!= i]
                    count1 += self.train_bs[i]

        self.num_sample_list.append(counting(self.train_bs, self.active_client, retain))
        self.active_client = retain
            
    def observe(self,round):
        self.norm_l = observe_record(self.X, self.server.x, self.lr_latest, round,\
                                      self.threshold_l, self.norm_l)
    def observe_save(self):
        cur_path = os.path.abspath(os.curdir)
        summary_path0 = os.path.join(cur_path,'results/{}'.format(self.dataset),\
                                     'data_{}_{}_{}'.format(self.alpha,self.beta0,self.iid),\
                                        'observations')
        os.makedirs(summary_path0, exist_ok=True)
        summary_path = os.path.join(summary_path0,\
                                'obs_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(self.algorithm,self.seed,self.seed1,self.seed2,self.bs,self.lr,self.global_round,self.local_round, self.K,self.epsilon, self.Beta, self.balanced))
        with open(summary_path,'w') as f:
            json.dump([self.norm_l, self.num_sample_list[0]], f)
    
    def summary_save(self):
        cur_path = os.path.abspath(os.curdir)
        summary_path0 = os.path.join(cur_path,'results/{}'.format(self.dataset),'data_{}_{}_{}'.format(self.alpha,self.beta0,self.iid),'data')
        os.makedirs(summary_path0, exist_ok=True)
        summary_path = os.path.join(summary_path0,\
                                'alg_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(self.algorithm,self.seed,self.seed1,self.seed2,self.bs,self.lr,self.global_round,self.local_round,self.K,self.epsilon,self.Beta,self.balanced))

        with open(summary_path,'w') as f:
            json.dump(self.server.tb, f)
        
        summary_path01 = os.path.join(cur_path,'results/{}'.format(self.dataset),'data_{}_{}_{}'.format(self.alpha,self.beta0,self.iid),'sample')
        
        os.makedirs(summary_path01, exist_ok=True)

        summary_path1 = os.path.join(summary_path01,\
                                'sample_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(self.algorithm,self.seed,self.seed1,self.seed2,self.bs,self.lr,self.global_round,self.local_round, self.K,self.epsilon, self.Beta, self.balanced))

        with open(summary_path1,'w') as f1:
            json.dump(self.num_sample_list, f1)
    
    def lr_update(self,round):
        if self.lr_decay:
            self.lr_latest = self.lr / np.sqrt(round+1)

    @abstractmethod
    def train(self):
        pass

#### trainer import ######
from .fedavg_trainer import fedavg_trainer
from .fedlavg_trainer import fedlavg_trainer
from .mifa_trainer import mifa_trainer
from .gm_trainer import gm_trainer
from .cclip_trainer import cclip_trainer


########## trainer selection #################

def trainer_select(config, algorithm, **kwargs):
    assert algorithm in ['FedAvg','FedProx','FedAvg_v','FedProx_v',\
                         'MIFA','gm','bucketing_gm','cclip','bucketing_cclip'],\
                            'Algorithm not supported!'
    if algorithm in ['FedAvg','FedProx']:
        trainer = fedavg_trainer(config,**config)
    elif algorithm in ['FedAvg_v','FedProx_v']:
        trainer = fedlavg_trainer(config,**config)
    elif algorithm in ['MIFA']:
        trainer = mifa_trainer(config,**config)
    elif algorithm in ['gm','bucketing_gm']:
        trainer = gm_trainer(config,**config)
    elif algorithm in ['cclip','bucketing_cclip']:
        trainer = cclip_trainer(config,**config)
        
    return trainer