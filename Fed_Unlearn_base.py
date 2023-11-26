# -*- coding: utf-8 -*-

import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import time

from FL_base import test,test_target

from FL_base import fedavg, global_train_once, FL_Train, median, trim_mean
import tqdm
import os


def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    
    all_start_time = time.time()

    my_index = 3

    '''step0. Set up the total save folder for the training model, check it, and create it if it does not exist'''
    print(5*"#"+f" {my_index}.0 Initialize Federated models save path "+5*"#")
    FL_params.chosen_clients_str = ""
    for _ in FL_params.selected_clients:
        FL_params.chosen_clients_str += "_" + str(_)
    FL_Model_dir = f"./{FL_params.model_result_name}/{FL_params.data_name}_clients{FL_params.chosen_clients_str}"
    FL_params.FL_Model_dir = FL_Model_dir
    
    if not os.path.exists(FL_Model_dir):
        os.makedirs(FL_Model_dir)
        print(5*"#"+f" {my_index}.0 Not found resutl dir, already made it: ",FL_Model_dir,5*"#")
    else:
        print(5*"#"+f" {my_index}.0 resutl dir is already existed: ",FL_Model_dir,5*"#")


    FL_params.attack_clients_str = ""
    for _ in FL_params.attack_clients_idx:
        FL_params.attack_clients_str += "_" + str(FL_params.selected_clients[_])

    

    '''step1. Train or import the initial federated learning model'''
    print(5*"#"+f" {my_index}.1 Federated Learning Start "+5*"#")
    FL_Primitive_Model_old_GMs_save_path = f'{FL_Model_dir}/FL_{FL_params.aggregation_method}_PGMs_Gp{FL_params.global_epoch}_Lp{FL_params.local_epoch}_bs{FL_params.local_batch_size}.pth'
    FL_Primitive_Model_old_CMs_save_path = f'{FL_Model_dir}/FL_{FL_params.aggregation_method}_PCMs_Gp{FL_params.global_epoch}_Lp{FL_params.local_epoch}_bs{FL_params.local_batch_size}.pth'
    std_time = time.time()
    if os.path.exists(FL_Primitive_Model_old_GMs_save_path) and os.path.exists(FL_Primitive_Model_old_CMs_save_path) and not FL_params.if_train:
        # Extract parameter
        old_GMs_dicts = torch.load(FL_Primitive_Model_old_GMs_save_path)
        old_CMs_dicts = torch.load(FL_Primitive_Model_old_CMs_save_path)
        # Initialization model
        old_GMs = [copy.deepcopy(init_global_model) for _ in range(len(old_GMs_dicts))]
        old_CMs = [copy.deepcopy(init_global_model) for _ in range(len(old_CMs_dicts))]
        # Load parameter
        for GMs_idx in range(len(old_GMs_dicts)):
            old_GMs[GMs_idx].load_state_dict(old_GMs_dicts[GMs_idx])
        for CMs_idx in range(len(old_CMs_dicts)):
            old_CMs[CMs_idx].load_state_dict(old_CMs_dicts[CMs_idx])

        old_epoch_result_in_test = np.load(FL_Primitive_Model_old_GMs_save_path.replace(".pth","_Rtest.npy"),allow_pickle=True).tolist()
        old_epoch_result_in_target = np.load(FL_Primitive_Model_old_GMs_save_path.replace(".pth","_Rtarget.npy"),allow_pickle=True).tolist()

        print(5*"#"+f" {my_index}.1 Find FL_Primitive_Model old_GMS & old_CMs, Load Successfuly! "+5*"#")
        
    else:
        # Training FL model
        old_GMs, old_CMs,  old_epoch_result_in_test, old_epoch_result_in_target = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
        # Extract parameter
        old_GMs_dicts = [model_temp.state_dict() for model_temp in old_GMs]
        old_CMs_dicts = [model_temp.state_dict() for model_temp in old_CMs]
        # Save parameter
        np.save(FL_Primitive_Model_old_GMs_save_path.replace(".pth","_Rtest.npy"),np.array(old_epoch_result_in_test,dtype=object))
        np.save(FL_Primitive_Model_old_GMs_save_path.replace(".pth","_Rtarget.npy"),np.array(old_epoch_result_in_target,dtype=object))
        torch.save(old_GMs_dicts, FL_Primitive_Model_old_GMs_save_path)
        torch.save(old_CMs_dicts, FL_Primitive_Model_old_CMs_save_path)
        
        print(5*"#"+f" {my_index}.1 Not Find FL_Primitive_Model old_GMS & old_CMs, Train and Save Successfuly! "+5*"#")
    
    end_time = time.time()
    time_learn = ( end_time - std_time )


    
    '''step2. Set the total save folder of the unlearning attack model, and check it, and create it if it does not exist'''
    print(5*"#"+f" {my_index}.2 Initialize Federated Unlearning Attack models save path "+5*"#")
    FL_Attacked_Model_dir = FL_Model_dir + f"/FL_{FL_params.aggregation_method}_Attaker_client{FL_params.attack_clients_str}"
    if not os.path.exists(FL_Attacked_Model_dir):
        os.makedirs(FL_Attacked_Model_dir)
        print(5*"#"+f" {my_index}.2 Not found resutl dir, already made it: ",FL_Attacked_Model_dir,5*"#")
    else:
        print(5*"#"+f" {my_index}.2 resutl dir is already existed: ",FL_Attacked_Model_dir,5*"#")
    FL_params.FL_Attacked_Model_dir = FL_Attacked_Model_dir
    


    # Calculate the output value of the target for subsequent attacks
    model_test = copy.deepcopy(old_GMs[-1])
    model_test.eval()
    FL_params.target_output = model_test(FL_params.target[0].unsqueeze(0)).squeeze().tolist()



    '''step3. Train an unlearn model under attack'''
    unlearn_attack_GMs_save_path = f'{FL_Attacked_Model_dir}/Target_{FL_params.target_idx}_UAGMs_{FL_params.attack_scheme}_itv{FL_params.unlearn_interval}_Fratio{str(int(FL_params.forget_local_epoch_ratio*100))}.pth'
    unlearn_attack_CMs_save_path = f'{FL_Attacked_Model_dir}/Target_{FL_params.target_idx}_UACMs_{FL_params.attack_scheme}_itv{FL_params.unlearn_interval}_Fratio{str(int(FL_params.forget_local_epoch_ratio*100))}.pth'
    print(5*"#"+f" {my_index}.3 Federated Unlearning Attack Start "+5*"#")
    std_time = time.time()
    if os.path.exists(unlearn_attack_GMs_save_path) and not FL_params.if_unlearning_attack:
        
        # Extract parameter
        unlearn_attack_GMs_dicts = torch.load(unlearn_attack_GMs_save_path)
        unlearn_attack_CMs_dicts = torch.load(unlearn_attack_CMs_save_path)
        # Initialization model
        unlearn_attack_GMs = [copy.deepcopy(init_global_model) for _ in range(len(unlearn_attack_GMs_dicts))]
        unlearn_attack_CMs = [copy.deepcopy(init_global_model) for _ in range(len(unlearn_attack_CMs_dicts))]
        # Load parameter
        for GMs_idx in range(len(unlearn_attack_GMs_dicts)):
            unlearn_attack_GMs[GMs_idx].load_state_dict(unlearn_attack_GMs_dicts[GMs_idx])
        for CMs_idx in range(len(unlearn_attack_CMs_dicts)):
            unlearn_attack_CMs[CMs_idx].load_state_dict(unlearn_attack_CMs_dicts[CMs_idx])

        attack_epoch_result_in_test = np.load(unlearn_attack_GMs_save_path.replace(".pth","_Rtest.npy"),allow_pickle=True).tolist()
        attakc_epoch_result_in_target = np.load(unlearn_attack_GMs_save_path.replace(".pth","_Rtarget.npy"),allow_pickle=True).tolist()
        all_change_result_dict_list = np.load(unlearn_attack_GMs_save_path.replace(".pth","_Rchange.npy"),allow_pickle=True).tolist()

        print(5*"#"+f" {my_index}.2 Find unlearn_attack_GMs, Load Successfuly! "+5*"#")
        
    else:
        # Training FL unlearning attack model
        unlearn_attack_GMs, unlearn_attack_CMs,  attack_epoch_result_in_test, attakc_epoch_result_in_target, all_change_result_dict_list = unlearning_attack(old_GMs, old_CMs, client_loaders, test_loader, FL_params)
        # Extract parameter
        unlearn_attack_GMs_dicts = [model_temp.state_dict() for model_temp in unlearn_attack_GMs]
        unlearn_attack_CMs_dicts = [model_temp.state_dict() for model_temp in unlearn_attack_CMs]
        # Save parameter
        np.save(unlearn_attack_GMs_save_path.replace(".pth","_Rtest.npy"),np.array(attack_epoch_result_in_test,dtype=object))
        np.save(unlearn_attack_GMs_save_path.replace(".pth","_Rtarget.npy"),np.array(attakc_epoch_result_in_target,dtype=object))
        np.save(unlearn_attack_GMs_save_path.replace(".pth","_Rchange.npy"),np.array(all_change_result_dict_list,dtype=object))
        torch.save(unlearn_attack_GMs_dicts, unlearn_attack_GMs_save_path)
        torch.save(unlearn_attack_CMs_dicts, unlearn_attack_CMs_save_path)

        print(5*"#"+f" {my_index}.2 Not Find unlearn_attack_GMs, Train and Save Successfuly! "+5*"#")
    end_time = time.time()
    time_unlearn_attack = ( end_time - std_time )
    print(5*"#"+" unlearn_attack End "+5*"#")

    

    '''step4. Prepare the output'''
    train_str = f"{FL_params.data_name}_clients{FL_params.chosen_clients_str}_{FL_params.aggregation_method}_Gepoch{FL_params.global_epoch}_Lepoch{FL_params.local_epoch}_batchsize{FL_params.local_batch_size}"#训练参数
    unlearn_attack_str = f"Target_{FL_params.target_idx}_Attaker_client{FL_params.attack_clients_str}_{FL_params.attack_scheme}_intervalc{FL_params.unlearn_interval}_Fratio{str(int(FL_params.forget_local_epoch_ratio*100))}"#遗忘攻击参数
    result = ( (train_str, old_epoch_result_in_test[-1],[len(old_epoch_result_in_test),test_target(old_GMs[-1], test_loader, FL_params.target_idx)]),  (unlearn_attack_str, attack_epoch_result_in_test[-1],attakc_epoch_result_in_target[-1]))
    # print the total time spent
    print(5*"#"+f" Idx.{my_index} cost time: "+f"{time.time()-all_start_time}"+5*"#")
    # print the change of the target
    print("target in old model:","loss=","{:.2e}".format(result[0][2][1][0],2),", output=",[round(_,2) for _ in result[0][2][1][2][0]],", pred_right=",result[0][2][1][1]==100)
    print("target in unlearning attack model:","{:.2e}".format(result[1][2][1][0],2),", output=",[round(_,2) for _ in result[1][2][1][2][0]],", attack_success=",result[1][2][1][1]<100)


    return old_GMs, old_CMs, unlearn_attack_GMs, unlearn_attack_CMs_dicts, result, all_change_result_dict_list
    
    
def unlearning_attack(old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    """
    
    Parameters
    ----------
    old_global_models : list of DNN models
        In standard federated learning, all the global models from each round of training are saved.
    old_client_models : list of local client models
        In standard federated learning, the server collects all user models after each round of training.
    client_data_loaders : list of torch.utils.data.DataLoader
        This can be interpreted as each client user's own data, and each Dataloader corresponds to each user's data
    test_loader : torch.utils.data.DataLoader
        The loader for the test set used for testing
    FL_params : Argment()
        The parameter class used to set training parameters

    Returns
    -------
    forget_global_model : One DNN model that has the same structure but different parameters with global_moedel
        DESCRIPTION.

    """

    # Check whether the parameters are correct
    if(not(max(FL_params.attack_clients_idx) in range(FL_params.N_client) and min(FL_params.attack_clients_idx)>0)):
        raise ValueError('FL_params.attack_clients_idx is note assined correctly, attack_clients_idx should in {}'.format(range(FL_params.N_client)))
    if(FL_params.unlearn_interval == 0 or FL_params.unlearn_interval >FL_params.global_epoch):
        raise ValueError('FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')
    
    # Initialization model
    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)
    
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    
    GM_intv = np.arange(0,FL_params.global_epoch+1, FL_params.unlearn_interval, dtype=np.int16())
    CM_intv = GM_intv[:-1]
    
    selected_GMs = [old_global_models[ii] for ii in GM_intv]
    selected_CMs = [old_client_models[jj] for jj in CM_intv]
    
    
    # Initializes the global model with epoch 0
    epoch = 0
    unlearn_global_models = list()
    unlearn_client_models = list()
    unlearn_global_models.append(copy.deepcopy(selected_GMs[0]))

    epoch_result_in_test = list()
    epoch_result_in_target = list()
    epoch_result_in_test.append( [0,test(unlearn_global_models[-1], test_loader)] )
    epoch_result_in_target.append( [0,test_target(unlearn_global_models[-1], test_loader, FL_params.target_idx)] )


    # Prepare the data set to unlearning attack
    prepare_std_time = time.time()
    client_data_loaders_prepare = list()
    attack_clients_data_loader = list()
    all_change_result_dict_list = list()
    attack_scheme = FL_params.attack_scheme
    print("attack_scheme:",attack_scheme)

    for ii_idx in range(FL_params.N_client):
        client_data_loader_temp = copy.deepcopy(client_data_loaders[ii_idx])
        client_data_loaders_prepare.append(client_data_loader_temp) 
        
        if ii_idx in FL_params.attack_clients_idx:
            attack_clients_data_loader.append(client_data_loader_temp)

    # Modify the data of the attacked client
    client_data_loader_temp,all_change_result_dict_list = dataloader_change_for_attack(attack_clients_data_loader,
                                                            attack_scheme, FL_params, old_GMs[-1])

    for i_idx,attack_client_idx in enumerate(FL_params.attack_clients_idx): 
        client_data_loaders_prepare[attack_client_idx] = copy.deepcopy(client_data_loader_temp[i_idx])

    print("attack dataloader preparation cost time:",time.time()-prepare_std_time)


    # Prepare unlearning epoch
    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    FL_params.global_epoch = CM_intv.shape[0]
    

    # Start training the unlearning model
    print('FL_Unlearning_Attack parameters: Clients = {} ,Global epoch = {}, Local Calibration Training epoch = {}'.format(FL_params.chosen_clients_str[1:],FL_params.global_epoch,FL_params.local_epoch))
    desc_str = f"{FL_params.aggregation_method},Attackers{FL_params.attack_clients_str},Target_{FL_params.target_idx}|FL Unlearning Attack:"
    global_epoch_bar = tqdm.trange(FL_params.global_epoch,colour="red",desc=desc_str)
    for epoch in global_epoch_bar:

        global_model = unlearn_global_models[epoch]

        new_client_models  = global_train_once(global_model, client_data_loaders_prepare, test_loader, FL_params)

        if epoch == 0:
            for ii_idx in range(FL_params.N_client):
                if ii_idx not in FL_params.attack_clients_idx:
                    new_client_models[ii_idx] = copy.deepcopy(selected_CMs[epoch][ii_idx])

        new_GM, new_CM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch], global_model, FL_params)
    

        unlearn_global_models.append(new_GM)
        unlearn_client_models += new_CM

        if FL_params.train_with_test:
            global_epoch_bar.set_postfix(A_L='{:.2e}'.format(epoch_result_in_test[-1][1][0]),# the loss of test set
                                     A_a='{:.1f}'.format(epoch_result_in_test[-1][1][1]),# the accuracy of test set
                                     T_L='{:.2e}'.format(epoch_result_in_target[-1][1][0]),# the loss of target
                                     T_a='{:.1f}'.format(epoch_result_in_target[-1][1][1]))# the accuracy of target
            
            epoch_result_in_test.append( [epoch+1,test(unlearn_global_models[-1], test_loader)] )
            epoch_result_in_target.append( [epoch+1,test_target(unlearn_global_models[-1], test_loader, FL_params.target_idx)] )
        
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch
    
    return unlearn_global_models, unlearn_client_models, epoch_result_in_test, epoch_result_in_target, all_change_result_dict_list



def Sample_Selection(X_train, y_train, X_target, y_target, model, hessian_inv_save_path, FL_params, criterion=nn.CrossEntropyLoss()):

    from torch.autograd.functional import hessian
    
    device = FL_params.device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_target = X_target.to(device)
    y_target = y_target.to(device)
    model.to(device)
    
    
    # select the influential samples for target
    loss_gain = []
    
    #grad for target
    X_target = X_target
    
    model.zero_grad()
    pred = model(X_target)
    y_target = y_target.type(torch.FloatTensor)
    loss = criterion(pred, y_target.to(device).long())
    loss.backward()
    
    grad_target = torch.cat(tuple([_.grad.view(-1) for _ in model.parameters()]),dim=0).view(-1)#获得梯度
    
    if os.path.exists(hessian_inv_save_path) and not FL_params.re_compute_influence:
        hessian_matrix_inv = torch.load(hessian_inv_save_path).to(device)
        print("Find hessian_inv in",hessian_inv_save_path,",load successfully")
        
    else:
        # Using torch.autograd.functional.hessian function to calculate Hessian Matrix
        hessian_matrix = hessian(lambda w:model.forward_loss(X_train,y_train,w), torch.cat(tuple([_.view(-1) for _ in model.parameters()]),dim=0))
        # the inverse 
        hessian_matrix = hessian_matrix+0.01*torch.eye(hessian_matrix.size(1), dtype=hessian_matrix.dtype, device=hessian_matrix.device)
        hessian_matrix_inv = hessian_matrix.inverse()


        torch.save(hessian_matrix_inv.cpu(),hessian_inv_save_path)
        if FL_params.re_compute_influence:
            print("Force Recompute hessian_inv, save in",hessian_inv_save_path)
        else:
            print("Not find hessian_inv, compute and save in",hessian_inv_save_path)

    
    for i in range(len(X_train)):
        
        x_train_i = X_train[i:i+1]
        y_train_i = y_train[i:i+1]
        #grad for each train sample
        model.zero_grad()
        pred = model(x_train_i)
        loss = criterion(pred, y_train_i)
        loss.backward()
        grad_x_train_i = torch.cat(tuple([_.grad.view(-1) for _ in model.parameters()]),dim=0).view(-1).clone()

        Delta1 = hessian_matrix_inv.mv(grad_x_train_i)
        
        loss_gain.append(-(grad_target*Delta1).sum().item())

    loss_gain = np.array(loss_gain)
    influence_sort = loss_gain.argsort()


    return influence_sort


def influence_sort_compute(client_dataloader_original,model,FL_params,client_str):

    data_tensor = torch.stack([sample[0] for sample in client_dataloader_original.dataset], dim=0)
    labels_tensor = torch.tensor([sample[1] for sample in client_dataloader_original.dataset])
    
    hessian_inv_save_path = f"{FL_params.FL_Model_dir}/Client_{client_str}_hessian_inv.pth"

    influence_sort = Sample_Selection(data_tensor, labels_tensor, FL_params.target[0].unsqueeze(0), 
                                          torch.tensor([FL_params.target[1]]), model, hessian_inv_save_path,
                                           FL_params, criterion=nn.CrossEntropyLoss())
    
    return influence_sort


def influence_data_change(all_client_dataloader_original,method,more_paraments,FL_params,model):

    number = int(more_paraments[0])
    result_dict = {}

    all_methods = ["feature"]
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # calculate the influence function
    all_influence_sorts = []
    for i_idx,client_idx in enumerate(FL_params.attack_clients_idx):
        client_str = str(FL_params.selected_clients[client_idx])
        # influence the point calculated by the function
        all_influence_sorts.append(influence_sort_compute(copy.deepcopy(all_client_dataloader_original[i_idx]),model,FL_params,client_str))

    
    if method == "feature":# Select the label with the second largest prediction probability of target to modify
        print(f'change the feature')
        # Maximum number of clients that can be selected
        single_client_max_num = int(np.ceil(int(more_paraments[1])/len(all_influence_sorts)))
        if single_client_max_num >= number:
            raise ValueError('single_client_max_num is bigger than slice_influence number, please check it! ')
        
        all_sliced_influence_idx_lists = list()
        for ii_idx,influence_sort in enumerate(all_influence_sorts):
            # Set the number to be modified
            modify_num = min(number, len(influence_sort))
            # Intercept length
            sliced_influence = influence_sort[:modify_num]
            sliced_influence_idx_list = [ [] for _ in range(10)]
            for idx in sliced_influence:
                data = all_client_dataloader_original[ii_idx].dataset[idx]
                label = data[1]
                sliced_influence_idx_list[label].append(idx)

            all_sliced_influence_idx_lists.append(sliced_influence_idx_list)
            
        
        # Select the modified sample points based on the predicted output of target
        target_output = np.array(FL_params.target_output)# output of target
        print("output is ",target_output)
        target_output[FL_params.target[1]] = min(target_output)-10# The label corresponding to target is changed to 10 less than the minimum value for easy sorting
        all_num_list_sort = np.argsort(-target_output)# The descending order of negative values is the order of positive values
        out_dataloader = list()
        out_result = list()
        for ii_idx,sliced_influence_idx_list in enumerate(all_sliced_influence_idx_lists):
            
            # Find the index of the sample you want
            modify_indices = list()
            change_class_num = list()
            for num_list_sort_max in all_num_list_sort:
                if len(modify_indices) >= single_client_max_num:
                        break
                
                label_num = 0
                for idx in sliced_influence_idx_list[num_list_sort_max]:
                    modify_indices.append(idx)
                    label_num += 1

                    if len(modify_indices) >= single_client_max_num:
                        break
                # Record the modification result of each client
                change_class_num.append( (str(num_list_sort_max),label_num) )

            dataset_new = CustomDataset(all_client_dataloader_original[ii_idx].dataset, modify_indices, FL_params.target[0])
            out_dataloader.append(DataLoader(dataset_new, all_client_dataloader_original[ii_idx].batch_size, shuffle=True, **kwargs))                        

            result_dict = {"change_index":modify_indices,"change_class_num":change_class_num}
            out_result.append(copy.deepcopy([str(FL_params.selected_clients[FL_params.attack_clients_idx[ii_idx]]),result_dict]))

            print("client_idx:",out_result[-1][0],"-----","change number:",result_dict["change_class_num"],"-----","total number:",len(result_dict["change_index"]))


        return out_dataloader, out_result


        
    else:
        raise ValueError(f'No such method in influence, please check it! Only {all_methods}')
    



def random_data_change(all_client_dataloader_original,method,more_paraments,FL_params,model):


    number = int(more_paraments[0])
    result_dict = {}

    all_methods = ["feature"]
    kwargs = {'num_workers': 0, 'pin_memory': True}
    ''''''
    

    if method == "feature":# Select the label with the second largest prediction probability of target to modify
        print(f'change the feature')
        # Maximum number of clients that can be selected
        single_client_max_num = int(number/len(FL_params.attack_clients_idx))
 
        # Generate different random sequences
        all_ranodm_idx_lists = list()
        for i_idx,client_idx in enumerate(FL_params.attack_clients_idx):

            old_index = np.array(all_client_dataloader_original[i_idx].dataset.indices)
            
            ranodm_idx_list = np.random.choice(np.arange(0, len(old_index)), size=len(old_index), replace=False)

            all_ranodm_idx_lists.append(ranodm_idx_list)
            
        
        # The number of changes per client is evenly distributed according to a random sequence
        out_dataloader = list()
        out_result = list()
        for ii_idx,ranodm_idx_list in enumerate(all_ranodm_idx_lists):
            
            # Find the index of the sample you want
            modify_indices = list()
            change_class_num = list()

            label_num_list = [0]*10
            for idx in ranodm_idx_list:
                data = all_client_dataloader_original[ii_idx].dataset[idx]
                label = data[1]
                label_num_list[label] = label_num_list[label] + 1
                modify_indices.append(idx)
                
                if len(modify_indices) >= single_client_max_num:
                        break
                
            # Record the modification result of each client
            for label_class,label_num in enumerate(label_num_list):
                if label_num>0:
                    change_class_num.append( (str(label_class),label_num) )

            dataset_new = CustomDataset(all_client_dataloader_original[ii_idx].dataset, modify_indices, FL_params.target[0])
            out_dataloader.append(DataLoader(dataset_new, all_client_dataloader_original[ii_idx].batch_size, shuffle=True, **kwargs))                        

            result_dict = {"change_index":modify_indices,"change_class_num":change_class_num}
            out_result.append(copy.deepcopy([str(FL_params.selected_clients[FL_params.attack_clients_idx[ii_idx]]),result_dict]))

            print("client_idx:",out_result[-1][0],"-----","change number:",result_dict["change_class_num"],"-----","total number:",len(result_dict["change_index"]))


        return out_dataloader, out_result

    else:
        raise ValueError(f'No such method in random, please check it! Only {all_methods}')
    



def dataloader_change_for_attack(client_dataloader_original,attack_scheme,FL_params,model):
    
    all_attack_schemes = ["Influence", "Random"]
    
    attack_scheme_split = attack_scheme.split("_")
    # The type is str, which is divided into four blocks: "Basic method _ Modification method _ Modification number _ Other parameters"
    # For the selection of influence function points, the modification scheme is divided into three parts: basic method _ modification method _ number of pre-selected points of influence function _ number of modification points
    # For random selection of points, the modification scheme is divided into three parts: basic method _ modification method _ number of modification points


    if attack_scheme_split[0]=="Influence":
        return influence_data_change(client_dataloader_original,attack_scheme_split[1],attack_scheme_split[2:],
                                     FL_params,model)
    
    elif attack_scheme_split[0]=="Random":
        return random_data_change(client_dataloader_original,attack_scheme_split[1],attack_scheme_split[2:],
                                     FL_params,model)
    
    else:
        raise ValueError(f'No such attack_scheme, please check it! Only {all_attack_schemes}')



def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget, FL_params):
    """
    
    Parameters
    ----------
    old_client_models : list of DNN models
        When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
        Models that require forgotten users are not discarded in the Forget function
    ref_client_models : list of DNN models
        When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
        Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model
        
    global_model_before_forget : The old global model
        DESCRIPTION.
    global_model_after_forget : The New global model
        DESCRIPTION.

    Returns
    -------
    return_global_model : After one iteration, the new global model under the forgetting setting

    """
    old_param_update = dict()#Model Params： oldCM - oldGM_t
    new_param_update = dict()#Model Params： newCM - newGM_t
    
    new_global_model_state = global_model_after_forget.state_dict()#newGM_t
    
    return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    

    assert len(old_client_models) == len(new_client_models)
    

    clients_dicts_list = list()
    for client_idx in range(len(old_client_models)):
        
        return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
        
        for layer in global_model_before_forget.state_dict().keys():
            old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
            new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
            
            return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer]
            
            # Copy old and new models
            old_param_update[layer] += old_client_models[client_idx].state_dict()[layer]
            new_param_update[layer] += new_client_models[client_idx].state_dict()[layer]
            
            # Calculate new and old update values
            old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]#Parameters: oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]#Parameters: newCM - newGM_t
            
            # Calculate the final update model
            step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
            step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
            
            return_model_state[layer] = new_global_model_state[layer] + step_length*step_direction
        
        clients_dicts_list.append( copy.deepcopy(return_model_state) )
    
    assert len(clients_dicts_list) == len(new_client_models)
    
    client_model = list()
    for client_dicts in clients_dicts_list:
        model_temp = copy.deepcopy(global_model_after_forget)
        model_temp.load_state_dict(client_dicts)
        client_model.append(model_temp)

    all_methods = ["fedavg", "median", "trim-mean"]
    if FL_params.aggregation_method == "fedavg":
        return fedavg(client_model), client_model
    elif FL_params.aggregation_method == "median":
        return median(client_model), client_model
    elif FL_params.aggregation_method == "trim-mean":
        return trim_mean(client_model, FL_params.percentage), client_model
    else:
        raise ValueError(f'No such method in aggregation, please check it! Only {all_methods}')
    
    


class CustomDataset(Dataset):
    def __init__(self, dataset, modify_indices, new_feature, transform=None):
        self.dataset = dataset
        self.modify_indices = modify_indices  # Index list of data to modify
        self.new_feature = new_feature          # New list of data values
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get raw MNIST data
        image, label = self.dataset[index]

        # Check if you need to modify the current data
        if index in self.modify_indices:
            # Get a new value, note that this is an example, you can modify it as needed
            new_image = self.new_feature
            image = new_image

        return image, label
    
    