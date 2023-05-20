
import argparse
import os
import json
import numpy as np

def get_config():
    parser = argparse.ArgumentParser(description='Example')
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help = 'dataset: synthetic, cifar10')
    parser.add_argument('--datapath', type=str, default='./data/', help='default loading dataset')
    # common parameters:
    parser.add_argument('--verbose', type=int, default=0, help = 'print out or not')
    parser.add_argument('--print_every', type=int, default=2, help='print out frequency')
    # observation related
    parser.add_argument('--threshold_l', type=str,\
                        default='5,10,15,50,100,150,200',\
                       help = 'threshold lists to make observations.')
    parser.add_argument('--observation', type=int, default=0, help = 'observation indicator')
    # synthetic data
    parser.add_argument('--alpha',type=str,default='0.1', help = 'synthetic param alpha')
    parser.add_argument('--beta0',type=float,default=1, help = 'synthetic param beta')
    parser.add_argument('--iid', type =float, default=0, help = 'synthetic param iid')
    # dropout param
    parser.add_argument('--epsilon',type=float,default=0.9, help = 'epsilon')
    parser.add_argument('--balanced', type=int, default=1, help = 'balanced local batch size: \
        each client with a batch size of "--bs", imbalanced: full batch (power law in the case of \
        synthetic data).')
    # learning
    parser.add_argument('--algorithm', type=str, default='FedLAvg', help = "algorithms in\
                ['FedAvg','FedLAvg','FedProx_step','FedProx_LAvg_step','MIFA']")
    parser.add_argument('--bs',type=int,default=100, help = 'batch size')
    parser.add_argument('--lr',type=float,default=0.1, help='learning rate')
    parser.add_argument('--lr_decay',type=int,default=0,help='learning rate decay')
    parser.add_argument('--wd',type=float,default=0, help='weight decay')
    parser.add_argument('--global_round',type=int,default=3000, help = 'communication round')
    parser.add_argument('--local_round',type=int, default=25, help = 'local steps')
    parser.add_argument('--K',type=int, default= 10, help = 'sample clients')
    parser.add_argument('--M',type=int, default= 100, help = 'number of clients')
    # FedProx
    parser.add_argument('--mu',type=float,default=0.1, help = 'Proximal term in FedProx')
    # seed
    parser.add_argument('--seed',type=int, default=777, help = 'random seed')
    parser.add_argument('--seed1',type=int, default=777, help = 'random seed for mean generation')
    parser.add_argument('--seed2',type=int, default=777, help = 'random seed for data generation')
    # Beta
    parser.add_argument('--Beta', type=float, default=10., help='global stepsize')
    # observation
    parser.add_argument('--T1', type=int, default=5, help='shadow round 1')
    parser.add_argument('--T2', type=int, default=150, help='shadow round 1')
    parser.add_argument('--K1', type=int, default=25, help='the size of candidate non-responsive set 1')
    parser.add_argument('--K2', type=int, default=10, help='the size of candidate non-responsive set 2')
    parser.add_argument('--intermediate', type=int, default=1, help='binary indicator for building C1 and C2')

    args, _ = parser.parse_known_args()
    args_dict = args.__dict__

    if args_dict['observation'] == 1:
        args_dict['K'] = args_dict['M']
        args_dict['print_every'] = 1
        
    return args_dict



def load_local_data(algorithm, seed1, seed2,alpha, beta0, iid, bs, balanced, dataset, seed, lr, global_round, local_round, K, epsilon, Beta,  **kwargs):
    # define the paths (train and test data)
    cur_path = os.path.abspath(os.curdir)
    file_name ='obs_{}_seed_{}_seed1_{}_seed2_{}_bs_{}_lr_{}_gr_{}_lround_{}_K_{}_epsilon_{}_beta_{}_balanced_{}.json'.format(algorithm,seed,seed1,seed2,bs,lr,global_round,local_round, K, epsilon, Beta, balanced)
    train_path = os.path.join(cur_path,'results/{}'.format(dataset),'data_{}_{}_{}/observations'.format(alpha,beta0,iid), file_name)
    # Open the train JSON file
    with open(train_path) as f:
        # Load the data from the file
        train_data = json.load(f)
    return train_data

if __name__ == '__main__':
    config = get_config()
    if config['intermediate']:
        data = load_local_data(**config)
        threshold_l = np.array([int(item) for item in config['threshold_l'].split(',')])
        T1_ind = np.where(threshold_l == config['T1'])[0][0]
        T2_ind = np.where(threshold_l == config['T2'])[0][0]
        diff_col = np.abs(np.array(data[0][T2_ind]) - np.array(data[0][T1_ind]))
        # descending order
        sorted_diff_ind = np.argsort(-diff_col)
        sorted_diff = np.sort(-diff_col)

        file_path = 'index_list_{}_{}_{}.json'.format(config['intermediate'],config['alpha'],config['algorithm'])
        with open(file_path,'w') as f:
            json.dump(sorted_diff_ind.tolist(),f)
    else:
        
        train_path = 'index_list_{}_{}_{}.json'.format(1,config['alpha'],'FedAvg')
        assert open(train_path), 'File not exists, please run the intermediate shadow experiments for FedAvg!'

        with open(train_path) as f:
            # Load the data from the file
            train_data = json.load(f)

        remain = train_data[:config['K1']]

        train_path = 'index_list_{}_{}_{}.json'.format(1,config['alpha'],'cclip')
        assert open(train_path), 'File not exists, please run the intermediate shadow experiments for cclip!'
        with open(train_path) as f:
            # Load the data from the file
            train_data1 = json.load(f)

        count = 0 
        for i in train_data1:
            if i not in remain and count < config['K2']:
                remain.append(i)
                count += 1
            else:
                continue
        
        print('>>> set built. saving in progress.')

        file_path = 'index_list_{}.json'.format(config['alpha'])
        with open(file_path,'w') as f:
            json.dump(remain,f)