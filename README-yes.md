# Unlearning attack in FedEraser (Federated Unlearning attack)
## About The Project
FedEraser allows a federated client to quit the Federated Learning system and eliminate the influences of his or her data on the global model trained by the standard Federated Learning. 
This project is based on federase to carry out amnesia attack, aiming to realize the attack on a specific target through the amnesia request of a specific client.

## Presented Unlearning Methods
**FedEraser (Federated Unlearning, which is named FedEraser in our paper)**.The parameters of the client model saved by the not forgotten user in the standard FL training process were taken as the step size of the global model iteration, and then the new global model was taken as the starting point for the training, and a small amount of training was carried out, and the parameters of the new Client model were taken as the direction of the iteration of the new global model.Iterate over the new global model using the $step \times direction$.

The main function is contained in Fed_Unlearn_main.py. 


## Getting Started
### Prerequisites
**Gradeint-Leaks** requires the following packages: 
- Python 3.8.3
- torch 2.0.0+cu118
- torchvision 0.15.1+cu118
- Sklearn 0.23.1
- Numpy 1.23.0
- Scipy 1.10.1


### File Structure 
```
Federated Unlearning attack in FedEraser
├── datasets
│   └── MNIST
├── result
├── data_preprocessing.py
├── Fed_Unlearn_base.py
├── Fed_Unlearn_main.py
├── FL_base.py
├── model_initiation.py
options.py
```
There are several parts of the code:
- datasets folder: This folder contains the training and testing data for the target model.  In order to reduce the memory space, we just list the  links to theset dataset here.      
   -- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz     
   -- Purchase: https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz    
   -- MNIST: http://yann.lecun.com/exdb/mnist/     
- data_preprocessing.py: This file contains the preprocessing of the raw data in datasets folder.
- Fed_Unlearn_base.py: This file contains the base function of FedEraser, which corresponds to **Section III** in our paper.
- ***Fed_Unlearn_main.py: The main function of Federated Unlearning attack.***      
- FL_base.py: This file contains the function of Federated Learning, such as FedAvg, Local-Training. 
- model_initiation.py: This file contains the structure of the global model corresponding to each dataset that we used in our experiment.  

## Parameter Setting of FedEraser
The attack settings of Federated Unlearning attack are determined in the parameter **FL_params** in **Fed_Unlearn_main.py** and **options.py**. 
- ***Federated Learning Model Training Settings***          
-- FL_params.data_name: the dataset name     
-- FL_params.data_split: the data set partitioning method      
-- FL_params.N_total_client: the number of federated clients         
-- FL_params.N_client: the number of selected federated clients      
-- FL_params.aggregation: aggregation method of federated learning      
-- FL_params.percentage: the remove percentage of each side in trim-mean aggregation method     
-- FL_params.global_epoch: the number of global training  epoch in federated learning     
-- FL_params.local_epoch: the number of client local training   epoch in federated learning     
-- FL_params.selected_clients: the list of selected clients for federated learning        


- ***Model Training Settings***     
-- FL_params.local_batch_size: the local batch size of the client's local training        
-- FL_params.local_lr: the local learning rate of the client's local training       
-- FL_params.test_batch_size: the testing  batch size of the client's local training      



- ***Federated Unlearning Settings***     
-- FL_params.unlearn_interval: Used to control how many rounds the model parameters are saved. $1$ represents the parameter saved once per round. (corresponding to N_itv in our paper)      
-- FL_params.forget_local_epoch_ratio: When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence. $forget_local_epoch_ratio \times local_epoch is$ the number of rounds of local training when we need to get the convergence direction of each local model                  



- ***Attack Settings***          
-- FL_params.attack_clients_idx: the clients that made the forget request to attack       
-- FL_params.target_idx: the index of the target sample that was attacked     
-- FL_params.target: the target sample    
-- FL_params.attack_scheme: the attack scheme      



- ***others***    
-- FL_params.seed: random seed      
-- FL_params.device: the designation of the operating device      
-- FL_params.model_result_name: the root path used to store training results     
-- FL_params.train_with_test: controlling whether testings are performed at the end of each global round of training    
-- FL_params.if_train: used to determine whether to import the model or retrain the complete federated learning model during the training phase. If true, it will be retrained every time. The default is false, and it will not be retrained when the model is found     
-- FL_params.if_unlearning_attack: used to determine whether to import the model or retrain the federated unlearning attack model during the training phase. If true, it will be retrained every time. The default is false, and it will not be retrained when the model is found         
-- FL_params.re_compute_influence: Used to determine whether the Hessian matrix of the client is recalculated during the calculation impact function phase. If true, it recalculates the Hessian matrix every time. The default value is FALSE, and the Hessian matrix is not retrained when it is found       

More explanation on the point selection scheme (controlled by parameter FL_params.attack_scheme):     
- The type is str, which is divided into four blocks: "Basic method _ Modification method _ Modification number _ Other parameters"      
- For the selection of influence function points, the modification scheme is divided into three parts: basic method _ modification method _ number of pre-selected points of influence function _ number of modification points    
-- For example：Influence_feature_300_60     
- For random selection of points, the modification scheme is divided into three parts: basic method _ modification method _ number of modification points     
-- For example：Random_feature_60      
- When multiple client attacks at the same time, the number of modification points is the total number of multiple client. In this paper, we use the average distribution of the number of points to be modified per client.


## Execute Federated Unlearning attack
The example given in this code is an attack against the mnist dataset   
>*** Run Fed_Unlearn_main.py. ***





