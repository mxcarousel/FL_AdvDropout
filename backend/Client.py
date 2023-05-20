from model import gen_model
from optimizer import FedProx
import torch.nn as nn
import numpy as np
import torch
import copy

class client:
    def __init__(self,train_loader,dataset,algorithm,lr,lr_decay,\
                 wd,mu,local_round,**kwargs):
        ########## parameters ########
        self.algorithm = algorithm
        self.local_round = local_round
        self.lr_decay = lr_decay
        self.lr = lr
        self.wd = wd
        self.mu = mu

        #########   dataset ##########
        self.train_loader = train_loader
        self.train_loader_iter = iter(train_loader)
        self.criterion = nn.CrossEntropyLoss()

        self.x = gen_model(dataset)

        if torch.cuda.is_available():
            self.x = self.x.cuda()

        assert algorithm in ['gm','bucketing_gm','cclip','bucketing_cclip',\
                             'MIFA','FedProx','FedAvg','FedProx_v','FedAvg_v'], \
                                'Algorithm not supported!'
        
        if algorithm in ['gm','bucketing_gm','cclip','bucketing_cclip']:
            self.optim = torch.optim.SGD(params=self.x.parameters(), lr=lr, momentum=0.9,\
                                          weight_decay=wd)
            self.local_round = 1
            self.m =  gen_model(dataset)
            self.bucket = None
            if algorithm in ['bucketing_gm','bucketing_cclip']:
                self.bucket = 2

        elif algorithm in ['MIFA']:
            self.optim = torch.optim.SGD(params=self.x.parameters(), lr=lr, momentum=0,\
                                          weight_decay=wd)
            # G0 is the $G_{t-1}^i$ in the paper;
            # G1 is the $\frac{1}{eta_t}(w_t - w_{t,K}^i)$ in the paper.
            self.G0 = gen_model(dataset)
            self.G1 = gen_model(dataset)
        
        elif algorithm in ['FedProx','FedProx_v']:
            self.optim = self.fedprox_optim_load()
        
        elif algorithm in ['FedAvg','FedAvg_v']:
            self.optim = torch.optim.SGD(params=self.x.parameters(), lr=lr, momentum=0, weight_decay=wd)

    def local(self,state_dict,lr):
        self.load(state_dict)
        self.x.train()
        ######## fedprox local solver ##########
        if self.algorithm in['FedProx','FedProx_v']:
            self.fedprox_optim_load()

        ####### general local trainings ########
        for _ in range(self.local_round):
            """"draw a fresh sample"""
            try:
                batch = self.train_loader_iter.__next__()
            except StopIteration:
                self.train_loader_iter = iter(self.train_loader)
                batch = self.train_loader_iter.__next__()
            if torch.cuda.is_available():
                data, target = batch[0].cuda(), batch[1].type(torch.LongTensor).cuda()
                self.x.cuda()
            else:
                data, target = batch[0], batch[1].type(torch.LongTensor)
            output = self.x(data)
            loss = self.criterion(output,target)
            loss.backward()
            if self.lr_decay:
                self.lr_update(lr)
            self.optim.step()
            self.optim.zero_grad()

    def lr_update(self,lr):
        for g in self.optim.param_groups:
            g['lr'] = lr

    def fedprox_optim_load(self):
        self.optim = FedProx(params = self.x.parameters(),lr=self.lr, momentum=0.9,\
                                     weight_decay = self.wd, mu = self.mu)

    def load(self,state_dict):
        self.x.load_state_dict(copy.deepcopy(state_dict))
    
    def mifa_update(self):
        self.G0.load_state_dict(copy.deepcopy(self.G1.state_dict()))

    def retrieve(self):
        return self.x.state_dict()
    
    
        
