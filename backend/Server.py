from model import gen_model
from utils import bucketing_wrapper,clip,smoothed_weiszfeld
import torch.nn as nn
import numpy as np
import datetime
import torch
import copy

class server:
    def __init__(self,train_loader,test_loader,train_bs,\
                 dataset,algorithm,lr,verbose,Beta,\
                    **kwargs):
        ################## dataset #########################
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_bs = train_bs
        ################## params #########################
        self.lr = lr
        self.dataset = dataset
        self.algorithm = algorithm
        self.Beta = Beta
        self.verbose = verbose
        ################## initialization ##################
        self.G = gen_model(dataset)
        self.criterion = nn.CrossEntropyLoss()

        self.tb = {'test_loss':[],'train loss':[],\
                    'test acc':[],'train acc':[],\
                    'epoch':[]}
        
        self.x = gen_model(dataset)

        if torch.cuda.is_available():
            self.x = self.x.cuda()
            self.G = self.G.cuda()
    
    def retrieve(self):
        return self.x.state_dict()

    def momentum_retrieve(self,active_client,X,round,lr):
        with torch.no_grad():
            for idx in active_client:
                for name, params in X[idx].m.named_parameters():
                    params.data = (self.x.state_dict()[name].data - X[idx].retrieve()[name].data) / lr  

    def fedavg_agg(self,active_client,X):
        with torch.no_grad():
            if torch.cuda.is_available():
                self.x.cuda()
            for name, param in self.x.named_parameters():
                param.data = torch.zeros_like(param.data)
                for idx in active_client:
                    param.data += X[idx].retrieve()[name].data \
                        * self.train_bs[idx]/ self.train_bs[active_client].sum()

    def fedlavg_agg(self,active_client,X):
        with torch.no_grad():
            if torch.cuda.is_available():
                self.x.cuda()
            for name, param in self.x.named_parameters():
                for idx in active_client:
                    param.data += self.Beta * ((X[idx].retrieve()[name].data -param.data) \
                                        * self.train_bs[idx]/ self.train_bs.sum())

    def mifa_agg(self,active_client,X,round,lr):
        for idx in active_client:
            with torch.no_grad():
                if torch.cuda.is_available():
                    self.x.cuda()
                    self.G.cuda()
                    X[idx].G0.cuda()
                    X[idx].G1.cuda()
                for name, param in X[idx].G1.named_parameters():
                    param.data = (self.x.state_dict()[name].data -\
                                X[idx].retrieve()[name].data) / lr
                for name, param in self.G.named_parameters():
                    param.data += (X[idx].G1.state_dict()[name].data - \
                                    X[idx].G0.state_dict()[name].data) \
                        * self.train_bs[idx] / self.train_bs.sum()
                X[idx].mifa_update()
        with torch.no_grad():
            for name, param in self.x.named_parameters():
                param.data -= self.G.state_dict()[name].data * lr
            
    def rfa_agg(self,active_client,X,round,lr):
        self.momentum_retrieve(active_client,X,round,lr)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        if X[0].bucket != None:
            m1, train_bs = bucketing_wrapper(X, self.train_bs, 2, len(X))
        else:
            m1 = [i.m.to(device) for i in X]
            train_bs = self.train_bs
        
        m_temp = copy.deepcopy(self.x).to(device)
        with torch.no_grad():
            for name, params in m_temp.named_parameters():
                weight_l = []
                for w in m1:
                    weight_l.append(w.state_dict()[name].data)
                params.data = smoothed_weiszfeld(weight_l, train_bs / train_bs.sum())
                
            for name, params in self.x.named_parameters():
                params.data = params.data - lr * m_temp.state_dict()[name].data

    def cclip_agg(self,active_client,X,round,lr):
        self.momentum_retrieve(active_client,X,round,lr)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        if X[0].bucket != None:
            m1, train_bs = bucketing_wrapper(X, self.train_bs, 2, len(X))
        else:
            m1 = [i.m.to(device) for i in X]
            train_bs = self.train_bs
        
        m_temp = copy.deepcopy(self.x).to(device)

        with torch.no_grad():
            for i in range(3):
                for name, params in m_temp.named_parameters():
                    if i == 0:
                        params = torch.zeros_like(params)
                    for idx, v in enumerate(m1):
                        params.data = clip(v.state_dict()[name].data - params.data) \
                        * train_bs[idx]/ train_bs.sum() + params.data
                
            for name, params in self.x.named_parameters():
                params.data = params.data - lr * m_temp.state_dict()[name].data

    def evaluate(self,epoch):
        if self.verbose:
            print(f"\n| Test All |", flush=True, end="")
        if torch.cuda.is_available():
            self.x.cuda()
        self.x.eval()
        total_loss, total_correct, total, step = 0, 0, 0, 0
        start = datetime.datetime.now()
        for batch in self.train_loader:
            step += 1
            if torch.cuda.is_available():
                data, target = batch[0].cuda(), batch[1].type(torch.LongTensor).cuda()
            else:
                data, target = batch[0], batch[1].type(torch.LongTensor)
            output = self.x(data)
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            end = datetime.datetime.now()
            if self.verbose:
                print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
        total_train_loss = total_loss / step
        total_train_acc = total_correct / total
        if self.verbose:
            print(f'\n| Test All Train Set |'
                f' communication round: {epoch},'
                f' loss: {total_train_loss:.4},'
                f' acc: {total_train_acc:.4%}', flush=True)
        total_loss, total_correct, total, step = 0, 0, 0, 0

        for batch in self.test_loader:
            step += 1
            if torch.cuda.is_available():
                data, target = batch[0].cuda(), batch[1].type(torch.LongTensor).cuda()
            else:
                data, target = batch[0], batch[1].type(torch.LongTensor)
            output = self.x(data)
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            end = datetime.datetime.now()
            if self.verbose:
                print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
        total_test_loss = total_loss / step
        total_test_acc = total_correct / total
        if self.verbose:
            print(f'\n Algorithm {self.algorithm}'
                f'\n| Test All Test Set |'
                f' communication round: {epoch},'
                f' loss: {total_test_loss:.4},'
                f' acc: {total_test_acc:.4%}', flush=True)
        self.tb['test_loss'].append(total_test_loss)
        self.tb['train loss'].append(total_train_loss)
        self.tb['test acc'].append(total_test_acc)
        self.tb['train acc'].append(total_train_acc)
        self.tb['epoch'].append(epoch)