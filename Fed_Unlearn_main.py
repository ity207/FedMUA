
import torch
import numpy as np


from model_initiation import mnist_model_init
from data_preprocess import data_init, data_init_non_iid

from Fed_Unlearn_base import federated_learning_unlearning
from options import args_parser


"""Step 0. Initialize Federated Unlearning parameters"""
class Arguments():
    def __init__(self, args):

        #Federated Learning Settings
        self.data_name = args.dataset
        self.data_split = args.data_split

        self.N_total_client = args.N_total_client
        self.N_client = args.N_client

        self.aggregation_method = args.aggregation  
        self.percentage = args.percentage        

        self.global_epoch = args.global_epoch  
        self.local_epoch = args.local_epoch

        self.selected_clients = [72,43,41,77,52,65,23,22,80,3]#np.random.choice(range(self.N_total_client),self.N_client,replace=False).tolist()
        
        

        #Model Training Settings
        self.local_batch_size = args.local_batch_size
        self.local_lr = args.local_learning_rate
        self.test_batch_size = args.local_batch_size
        

        
        #Federated Unlearning Settings
        self.unlearn_interval= args.unlearn_interval
            #Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
            #If this parameter is set to False, only the global model after the final training is completed is output        
        self.forget_local_epoch_ratio = args.forget_local_epoch_ratio 
            #When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
            #forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model



        # Attack Settings
        self.attack_clients_idx = args.attack_clients_idx
        self.target_idx = args.target_idx 
        self.target = None
        self.attack_scheme = args.attack_scheme


        
        # others
        self.device = args.device
        self.seed = args.seed
        self.model_result_name = args.result_dir
        self.train_with_test = True
        self.if_train = False   
        self.if_unlearning_attack = False    
        self.re_compute_influence = False




def Federated_Unlearning():

    """Step 1.Set the parameters for Federated Unlearning"""
    args = args_parser()
    FL_params = Arguments(args)
    print(args)
    
    torch.manual_seed(FL_params.seed)
    #kwargs for data loader 
    print(60*'=')
    print("Step1. Federated Learning Settings \n We use dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment.\n"))



    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")  
    init_global_model = mnist_model_init(FL_params.data_name)

    if FL_params.data_split == 'iid':
        client_all_loaders, test_loader = data_init(FL_params)
    elif FL_params.data_split == 'noniid':
        client_all_loaders, test_loader = data_init_non_iid(FL_params)
    else:
        raise ValueError(f'No such data_split, please check it! Only iid or noniid')

    client_loaders = list()
    all_train_num = 0
    for idx in FL_params.selected_clients:
        client_loaders.append(client_all_loaders[idx])
        all_train_num += len(client_all_loaders[idx].dataset)
    


    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60*'=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    # get the target data info
    FL_params.target = test_loader.dataset[FL_params.target_idx]
    

    old_GMs, old_CMs, unlearn_attack_GMs, unlearn_attack_CMs_dicts, result, all_change_result_dict_list = federated_learning_unlearning(init_global_model, 
                                                        client_loaders, 
                                                        test_loader, 
                                                        FL_params)
                
    print("All done!")

if __name__=='__main__':
    Federated_Unlearning()














































