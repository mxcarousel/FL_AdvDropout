# Federated Learning in the Presence of Adversarial Client Unavailability

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

This repository is an official implementation of the following paper:

>**Federated Learning in the Presence of Adversarial Client Unavailability**, submitted to *Neural Information Processing Systems (NeurIPS) 2023*.

## Requirements
The implementation runs on:
* Python 3.7.15
* PyTorch 1.13.1
* Torchvision 0.14.1

All experiments in this paper are conducted on a cluster with 8 Tesla V100 Volta GPUs and 80 Intel Xeon E5 CPUs.

## Data Generation
Before the executions of the actual experiments, we need to partition the global dataset (for CIFAR-10) or generate synthetic data. In addition, the shadow datasets are generated in advance using different random seeds from the actual experiments.

Below is an example to launch the data generation script.

```bash
python -m gen_data --dataset cifar10 --balanced 1 \
										--M 100 --alpha 0.1 --beta 1 \ 
										--iid 0 --train_num 50000 \ 
										--test_num 10000 --class_num 10 \
										--save 1 --seed1 777 --seed2 777
```
***Explanation of arguments:***

* ```dataset```: dataset name, either ```cifar10``` or ```synthetic```;
* ```balanced```: binary indicator for a balanced dataset or unbalanced dataset with clients' local volume following a power law;
* ```M```: number of clients;
* ```alpha```: Dirichlet allocation parameter for CIFAR-10; first non-IID parameter for synthetic dataset;
* ```beta```: second non-IID parameter for synthetic dataset;
* ```iid```: binary IID indicator for synthetic dataset;
* ```train_num```: total number of training examples in CIFAR-10 experiments;
* ```test_num```: total number of testing examples in CIFAR-10 experiments;
* ```class_num```: total number of classes in the dataset;
* ```save```: binary indicator for saving the generated partition or synthetic data;
* ```seed1```: random seed for CIFAR-10 partition or for synthetic $u_k,v_k,$ and $B_k$;
* ```seed2```: random seed for synthetic data generation.

##  Shadow Experiments

#### (i) Shadow datasets

For CIFAR-10 executions, we can generate a different partition of local datasets, or so-called  shadow local datasets, by choosing two different ```seed1```. 

```seed2``` is an argument for synthetic data generation only. 

Below is an example bash script for regular and shadow dataset partition generation.

```bash
# Regular dataset
python -m gen_data --dataset cifar10 --balanced 1 --alpha 0.1
# Shadow dataset
python -m gen_data --dataset cifar10 --balanced 1 --alpha 0.1 \
										--seed1 456
```

On the other hand, choosing different ```seed2``` while keeping ```seed1``` as the same yields different realizations of the same non-IID distribution of synthetic dataset. 

We also give an example on synthetic shadow dataset generation below.

```bash
# Regular dataset
python -m gen_data --dataset synthetic --balanced 0 \
										--alpha 1 --beta 1
# Shadow dataset
python -m gen_data --dataset synthetic --balanced 0 \
										--alpha 1 --beta 1 --seed2 456
```

#### (ii) Shadow executions

* As illustrated in the paper, the shadow executions are experiments with full-client participation but on the shadow datasets. Thus, we need to implement the experiments in **Launch Experiments** first but with the binary indicator ```observation``` set as 1. This will automatically change ```K=M```, i.e., full-client participation. Make sure $T_1,T_2 \in $ ```threshold_l```;
* What is more, to correctly load in the files, we need the correct specifications of seed1 and seed2 as we have discussed in (i);
* After the shadow experiments are done, run the following code in order which will automatically generate the candidate non-responsive set $\tilde{\cal{C}}$.

```bash
# First run the shadow executions
## shadow example for CIFAR-10
python -m main --algorithm FedAvg --dataset cifar10 --seed1 456 --observation 1
python -m main --algorithm cclip --dataset cifar10 --seed1 456 --observation 1
## alternatively, shadow example for synthetic
python -m main --algorithm FedAvg --dataset synthetic --alpha 1 --seed1 456 \
								--observation 1
python -m main --algorithm cclip --dataset synthetic --alpha 1 --seed1 456 \
								--observation 1

# Then collect the sorted client index from the files:
python -m gen_shadow --observation 1 --intermediate 1 --algorithm FedAvg --T1 5 --T2 150
python -m gen_shadow --observation 1 --intermediate 1 --algorithm cclip --T1 5 --T2 150

# Build $\tilde{\cal{C}}$ from $\tilde{\cal{C}}_1$ and $\tilde{\cal{C}}_2$:
python -m gen_shadow --observation 1 --intermediate 0 --K1 25 --K2 10
```
The other hyperparameters should align with the executions in the shadow experiments and can be found in **Launch Experiments**.

## Lauch Experiments

The experiment execution on Assumption 2 is a bit different from the paper. In particular, the experiments use $\sum_{i\in \tilde{\cal{S}}_t \setminus \cal{S}_t} < \epsilon KN/M$ rather than $\sum_{i\in \tilde{\cal{S}}_t \setminus \cal{S}_t} \le \epsilon KN/M$ in the paper. 

However, they are still equivalent by manipulating $\epsilon$. For example, $\epsilon = 0.9$ in the below code is equivalent to $\epsilon = 0.8 $ in the paper under balanced local datasets.

```bash
python -m main --dataset cifar10 --verbose 0 --print_every 1 \
								--observation 0 --threshold_l 0,5,150 --epsilon 0.9 \
								--alpha 0.1 --beta 1 --iid 0 --balanced 1 \
								--algorithm FedAvg --bs 100 --lr 0.1 --lr_decay 1 \
								--global_round 2000 --local_round 25 --K 10 \
								--M 100 --mu 0.1 --seed 777 --seed1 777 --seed2 777 \
                --Beta 10.
```
Some of the arguments are the same with the ones in data generation; however, for a clear and coherent presentation, we also reintroduce them here.
***Explanation of arguments:***

* ```dataset```: dataset name, either ```cifar10``` or ```synthetic```;
* ```verbose```: print out the training details (train loss, test accuracy) per evaluation round;
* ```print_every```: record frequency of the evaluation details;
* ```observation```: choose to conduct observation, or do the actual experiments;
* ```threshold_l```: observation recording communication round, i.e., $T_1$,$T_2$, and possibly $T_k$, where $k\ge 2$;
* ```epsilon```: dropout fraction as in Assumption 2;
* ```M```: number of clients;
* ```alpha```: Dirichlet allocation parameter for CIFAR-10; first non-IID parameter for synthetic dataset;
* ```beta0```: second non-IID parameter for synthetic dataset;
* ```iid```: binary IID indicator for synthetic dataset;
* ```balanced```: binary indicator for a balanced dataset or unbalanced dataset with clients' local volume following a power law;
* ```algorithm```: algorithm name used for a given experiments. The supported candidates are FedAvg, FedAvg_v, FedProx, FedProx_v, MIFA, cclip, bucketing_cclip, gm, bucketing_gm, where FedAvg_v and FedProx_v denotes the proposed variants, respectively;
* ```bs```: batch size. This argument is enables in FedAvg_type and can be interpreted as $n_i$ for $i\in[M]$;
* ```lr```: learning rate. Notably, this is also the learning rate for FedProx local program;
* ```lr_decay```: binary indicator for a decaying learning rate of non-convex function, i.e., $\eta_t = \eta_0 / \sqrt{t+1}$;
* ```global_round```: global communication round number;
* ```local_round```: local steps $s$;
* ```K```: the size of the set $\tilde{\cal{S}}_t$;
* ```M```: the size of the participating clients;
* ```mu```: the proximal coefficient;
* ```seed```: random seed in control of model initialization and client sampling per communication round;
* ```seed1```: random seed for CIFAR-10 partition or for synthetic $u_k,v_k,$ and $B_k$;
* ```seed2```: random seed for synthetic data generation;
* ```Beta```: amplification factor $\beta$.

## Results Visualization
We provide the interested readers with a Jupyter notebook, titled *'visualization.ipynb'*, to visualize the results.

There are two key functions:

##### Train loss and test accuracy visualization function

```
results_visualization_l(algorithms,configs,threshold, seed_list,smoothed)
```

***Explanation of arguments:***

* ```algorithms```: list of candidate algorithms;

* ```config```: list of candidate algorithms' configurations with the each item as the same structure of ```get_config()```;

* ```threshold```: visualize only part of the data points (up to threshold communication rounds);

* ```smoothed```: 0, do not skip any points; an integer, skip every integer number of data points.

##### Dropout fraction $\epsilon_t$ count visualization
```
dropout_plot(config,config1,seed_list,threshold)
```

***Explanation of arguments:***
* ```config```: FedAvg variant configurations;
* ```config1```: FedProx variant configurations;
* ```seed_list```: list of candidate random seeds;
* ```threshold```: visualize only part of the data points (up to threshold communication rounds).

  