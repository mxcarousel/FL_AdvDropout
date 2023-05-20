from .logistic import Logistic
from .lenet import LeNet

def gen_model(dataset):
    if dataset == 'synthetic':
        w = Logistic(60,10)
    elif dataset == 'cifar10':
        w = LeNet()
    return w