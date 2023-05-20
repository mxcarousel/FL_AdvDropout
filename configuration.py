import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Example')
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help = 'dataset: synthetic, cifar10')
    parser.add_argument('--datapath', type=str, default='./data/', help='default loading dataset')
    # common parameters:
    parser.add_argument('--verbose', type=int, default=0, help = 'print out or not')
    parser.add_argument('--print_every', type=int, default=1, help='print out frequency')
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

    
    args, _ = parser.parse_known_args()
    args_dict = args.__dict__

    if args_dict['observation'] == 1:
        args_dict['K'] = args_dict['M']

    return args_dict
