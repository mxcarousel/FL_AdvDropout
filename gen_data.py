import torch
import numpy as np
import argparse
import pickle
import json
import os
import random
from tqdm import trange
import torchvision

########################### Environmental Variables ##################################

def get_config():
    parser = argparse.ArgumentParser(description='Data Generation')

    parser.add_argument('--dataset',
                        default = 'cifar10',
                        type = str,
                        help = 'datasetname')
    parser.add_argument('--M',
                        default = 100,
                        type = int,
                        help = 'Number of users')
    parser.add_argument('--alpha',
                        default = '0.1',
                        type = str,
                        help = 'Dirichlet parameter or synthetic parameter #1')
    parser.add_argument('--beta',
                        default = '1',
                        type = str,
                        help = 'Synthetic parameter #2')
    parser.add_argument('--iid',
                        default = 0,
                        type = int,
                        help = 'Synthetic iid indicator')
    parser.add_argument('--train_num',
                        default = 50000,
                        type = int,
                        help = 'Number of train samples')
    parser.add_argument('--test_num',
                        default = 10000,
                        type = int,
                        help = 'Number of test samples')
    parser.add_argument('--class_num',
                        default = 10,
                        type = int,
                        help = 'Number of classes')
    parser.add_argument('--save',
                        default = 1,
                        type = int,
                        help = 'Storage indicator')
    parser.add_argument('--seed1',
                        default = 777,
                        type = int,
                        help = 'Random seeds for dataset partition (CIFAR-10) \
                            or for mean vectors u_i, v_i generation')
    parser.add_argument('--seed2',
                        default = 777,
                        type = int,
                        help = 'Random seeds for synthetic data generation')
    parser.add_argument('--balanced',
                        default = 1,
                        type = int,
                        help = 'Balanced data volume or imbalanced following a power law')

    args, _ = parser.parse_known_args()

    return args.__dict__

########################### Synthetic Data Generation ##############################

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, Beta, iid, M, seed1, seed2, balanced):
    
    np.random.seed(seed1)
    dimension = 60
    NUM_CLASS = 10

    if balanced:
        samples_per_user = 500 * np.ones(M).astype(int)
    else:
        samples_per_user = np.random.lognormal(4, 2, (M)).astype(int) + 50

    X_split = [[] for _ in range(M)]
    y_split = [[] for _ in range(M)]

    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, M)
    mean_b = mean_W
    B = np.random.normal(0, Beta, M)
    mean_x = np.zeros((M, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(M):
        mean_x[i] = np.random.normal(B[i], 1, dimension)

    np.random.seed(seed2)

    for i in range(M):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split

####################### CIFAR-10 Dirichlet Utensils ################################

class ImageDataset(object):
    def __init__(self, images, labels):
        if isinstance(images, torch.Tensor):
            self.data = images.numpy()
        else:
            self.data = images
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels
        self.label_count = []

    def __len__(self):
        return len(self.target)
    
    def Shuffle(self, class_num):
        train_example_indices = []
        for k in range(class_num):
            # Select all indices where the train label is k
            train_label_k = np.where(self.target == k)[0]
            np.random.shuffle(train_label_k)  
            train_example_indices.append(train_label_k)
            self.label_count.append(train_label_k.shape[0])
        print(self.label_count)
        return train_example_indices

def Gen_dirichlet(seed, alpha, M):
    train_multinomial_vals = []
    np.random.seed(seed)
    for i in range(M):
        proportion = np.random.dirichlet(alpha  * np.ones(10,))
        train_multinomial_vals.append(proportion)
    train_multinomial_vals =  np.array(train_multinomial_vals)
    return train_multinomial_vals


def main(dataset, seed1, seed2, iid, balanced, train_num, test_num, class_num,\
          M, alpha, beta, save, **kwargs):
    cpath = os.path.dirname(os.path.abspath(__file__))
    try:
        alpha = int(alpha)
    except ValueError:
        alpha = float(alpha)
    try:
        beta = int(beta)
    except ValueError:
        beta = float(beta)
    assert dataset in ['cifar10','synthetic'], 'dataset not supported!'
    if dataset == 'cifar10':
        dataset_dir = os.path.join(cpath, dataset)
    
        # Get CIFAR data, normalize, and divide by level
        print('>>> Get CIFAR10 data.')
        trainset = torchvision.datasets.CIFAR10(dataset_dir, download=True, train=True)
        testset = torchvision.datasets.CIFAR10(dataset_dir, download=True, train=False)

        train_d = ImageDataset(trainset.data, trainset.targets)
        test_d = ImageDataset(testset.data, testset.targets)
        
        train_example_indices = train_d.Shuffle(class_num)
        test_example_indices = test_d.Shuffle(class_num)

        train_count = np.zeros(10).astype(int)
        total_count_train = 0


        train_multinomial_vals = Gen_dirichlet(seed1, alpha, M)
        test_multinomial_vals = np.copy(train_multinomial_vals)
        if balanced:
            train_examples_per_user = [int(train_num / M) for _ in range(M)]
            test_examples_per_user = [int(test_num / M) for _ in range(M)]
        else:
            train_examples_per_user = np.random.lognormal(4, 2, (M)).astype(int) + 50
            test_examples_per_user = (train_examples_per_user / 5).astype(int)

        train_X = [[] for _ in range(M)]
        train_y = [[] for _ in range(M)]
        class_info = []

        for user in range(M):
            user_class_count = np.zeros(class_num).astype(int)
            for i in range(train_examples_per_user[user]):
                # count the number of training samples for each class
                sampled_label = np.argwhere(np.random.multinomial(1, train_multinomial_vals[user]) == 1)[0][0]
                current_sample = train_example_indices[sampled_label][train_count[sampled_label]]
                train_X[user].append(train_d.data[current_sample])
                train_y[user].append(sampled_label)
                train_count[sampled_label] += 1
                user_class_count[sampled_label] += 1
                total_count_train += 1
                if train_count[sampled_label] == train_d.label_count[sampled_label] and total_count_train < train_num:
                    train_multinomial_vals[:, sampled_label] = 0
                    train_multinomial_vals = train_multinomial_vals / np.sum(train_multinomial_vals, axis = 1)[:, None]
            print("the number of training samples held by user {}: {}".format(user, np.sum(user_class_count)))
            # return the class that the client has the most data samples
            class_info.append(np.argmax(user_class_count))

            total_count_test = 0
            test_count = np.zeros(class_num).astype(int)

        test_X = [[] for _ in range(M)]
        test_y = [[] for _ in range(M)]

        for user in range(M):
            user_class_count = np.zeros(class_num).astype(int)
            for i in range(test_examples_per_user[user]):
                # count the number of test samples for each class
                sampled_label = np.argwhere(np.random.multinomial(1, test_multinomial_vals[user]) == 1)[0][0]
                current_sample = test_example_indices[sampled_label][test_count[sampled_label]]
                test_X[user].append(test_d.data[current_sample])
                test_y[user].append(sampled_label)
                test_count[sampled_label] += 1
                user_class_count[sampled_label] += 1
                total_count_test += 1
                if test_count[sampled_label] == test_d.label_count[sampled_label] and total_count_test < test_num:
                    test_multinomial_vals[:, sampled_label] = 0
                    test_multinomial_vals = test_multinomial_vals / np.sum(test_multinomial_vals, axis = 1)[:, None]
            print("the number of test samples held by user {}: {}".format(user, np.sum(user_class_count)))


        # Setup directory for train/test data
        print('>>> Set data path for {}.'.format(dataset))
        train_path = '{}/{}/{}/{}/data/train/all.pkl'.format(cpath,'dataset',dataset,'dirichlet_' + str(alpha)+'_{}'.format(seed1))
        test_path = '{}/{}/{}/{}/data/test/all.pkl'.format(cpath,'dataset',dataset,'dirichlet_' + str(alpha)+'_{}'.format(seed1))
    
        dir_path = os.path.dirname(train_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        dir_path = os.path.dirname(test_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Create data structure
        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = dict()
        x_test, y_test, num_test = [], [], 0

        # Setup 100 users
        for i in range(M):
            uname = i
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
            train_data['num_samples'].append(len(train_X[i]))

            x_test.extend(test_X[i])
            y_test.extend(test_y[i])

            num_test += len(test_X[i])

        test_data['user_data'] = {'x': x_test, 'y': y_test}
        test_data['num_samples'] = num_test

        print('>>> User data distribution: {}'.format(train_data['num_samples']))
        print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
        print('>>> Total testing size: {}'.format(num_test))

        # Save user data
        if save:
            with open(train_path,'wb') as outfile:
                pickle.dump(train_data, outfile)
            with open(test_path, 'wb') as outfile:
                pickle.dump(test_data, outfile)

            print('>>> Save data.')

    elif dataset == 'synthetic':
        # The code was adapted from Tian Li's FedProx repository
        # https://github.com/litian96/FedProx
        
        cur_path = os.path.abspath(os.curdir)
        subfolder_train_path = 'dataset/synthetic/data_{}_{}_{}_{}_{}/train'.format(alpha,beta,iid, seed1, seed2)
        subfolder_test_path = 'dataset/synthetic/data_{}_{}_{}_{}_{}/test'.format(alpha,beta,iid, seed1, seed2)

        if not os.path.exists(subfolder_train_path):
            os.makedirs(subfolder_train_path)
        if not os.path.exists(subfolder_test_path):
            os.makedirs(subfolder_test_path)

        train_path = os.path.join(cur_path,subfolder_train_path, "mytrain.json")
        test_path = os.path.join(cur_path, subfolder_test_path, "mytest.json")

        X, y = generate_synthetic(alpha, beta, iid, M, seed1, seed2, balanced)     # synthetic (alpha,beta)

        # Create data structure
        train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
        test_data = dict()
        X_test, y_test, test_samples = [] , [], 0
        for i in trange(M, ncols=120):

            uname = 'f_{0:05d}'.format(i)        
            combined = list(zip(X[i], y[i]))
            random.shuffle(combined)
            X[i][:], y[i][:] = zip(*combined)
            num_samples = len(X[i])
            train_len = int(0.9 * num_samples) #train, test split, fraction 0.9/0.1
            test_len = num_samples - train_len
            
            train_data['users'].append(uname) 
            train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
            train_data['num_samples'].append(train_len)

            X_test.extend(X[i][train_len:])
            y_test.extend(y[i][train_len:])
            test_samples += test_len
            
        test_data['user_data'] = {'x': X_test, 'y': y_test}
        test_data['num_samples'] = test_samples
        if save:
            with open(train_path,'w') as outfile:
                json.dump(train_data, outfile)
            with open(test_path, 'w') as outfile:
                json.dump(test_data, outfile)
            print('>>> Save data.')

if __name__ == '__main__':
    config = get_config()
    main(**config)