
import numpy as np
import scipy
import torch
import utils
import random
import torch.optim as optim
import os
import model.net as net
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_observation_tensor(N_x, N_y, N_z, sampling_rate):
    N=N_x*N_y*N_z
    random_vector = np.random.choice([0, 1], size=N, replace=True, p=[1-sampling_rate,sampling_rate])
    return np.reshape(random_vector, (N_x, N_y, N_z))




def get_rmse(x,x_hat):
    return torch.sqrt(torch.mean( (x_hat - x)**2))







############################ data preprocessing##########################
# ## load ssf data
data = scipy.io.loadmat('./ssf_data/data.mat')
ssf = np.array(data['data']).astype('float64')
x_true = ssf

[N_x,N_y,N_z] = ssf.shape
mean = scipy.io.loadmat('./ssf_data/data_mean.mat')
data_mean = np.array(mean['data_mean']).astype('float64')

####### loading data
# choose sampling rate
sampling_rate = 0.1
data_sample_dir = './ssf_data/sampling_rate_'+str(sampling_rate)+".mat"
data_sample = scipy.io.loadmat(data_sample_dir)
observation_tensor=np.array(data_sample['observation_tensor']).astype('float64')
x_ob = ssf * observation_tensor

## normalize
ssf_max = np.max(np.abs(ssf))
x_ob_norm = x_ob/ssf_max


def train_and_evaluate(model,  optimizer, loss_fn,  x_ob, x_true, observation_tensor, TV=False):
    x_tensor = torch.FloatTensor(x_true).cuda()
    x_tensor = x_tensor.unsqueeze(0)
    
    x_ob_tensor = torch.FloatTensor(x_ob).cuda()
    x_ob_tensor = x_ob_tensor.unsqueeze(0)
    ot = torch.FloatTensor(observation_tensor).cuda()
    ot = ot.unsqueeze(0)


    rmse_min = 10
    train_loss_list=[]
    rmse_list=[]
    sparsity_list=[]
    time_tick=[]

    start_time = time.time()


    X_input = model.cut_tensor_into_sliding_patches(x_ob_tensor, subtensor_size=(model.subtensor_size, model.subtensor_size, model.subtensor_size), stride=model.stride)

    for epoch in tqdm(range(num_epochs)):
        output, _ = model(X_input)
        loss = loss_fn(output, x_ob_tensor, ot, add_TV_regu=TV)
        train_loss_list.append(loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if True:
            output_denorm = output.detach() * ssf_max
            x_tensor_true = x_tensor.detach()
            total_rmse = get_rmse(output_denorm, x_tensor_true)
            rmse_list.append(total_rmse.item())
            total_rmse = total_rmse.cpu().numpy()
            if total_rmse < rmse_min:
                rmse_min = total_rmse
                print("--------epoch:",epoch,"loss:",loss.item(),";min_rmse:", rmse_min)
    return rmse_min





def train_and_evaluate_tnn(model,  optimizer, loss_fn,  x_ob, x_true, observation_tensor,  TV=False):
    x_tensor = torch.FloatTensor(x_true).cuda()
    x_tensor =x_tensor.unsqueeze(0)
    #noise = 0 * np.random.randn(N_x, N_y, N_z)
    x_ob_tensor = torch.FloatTensor(x_ob).cuda()
    x_ob_tensor = x_ob_tensor.unsqueeze(0)
    ot = torch.FloatTensor(observation_tensor).cuda()
    ot = ot.unsqueeze(0)

    # grad_parameters = [param for param in model.parameters() if param.requires_grad]
    #
    # print("Parameters with Gradients:")
    # for param in grad_parameters:
    #     print(param)
    #     print(param.shape)

    total_rmse_min=10
    train_loss_list=[]
    rmse_list=[]
    relateive_error_list=[]
    time_tick=[]
    start_time = time.time()
    for epoch in tqdm(range(num_epochs_tnn)):


        output, core = model()
        loss = loss_fn(output, x_ob_tensor, ot, add_TV_regu=TV)
        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_denorm=output.detach()*ssf_max
        x_tensor_true=x_tensor.detach()


        total_rmse = get_rmse(output_denorm,x_tensor_true)
        rmse_list.append(total_rmse.item())
        total_rmse=total_rmse.cpu().numpy()

        if epoch % 1000 == 0:
            print("loss:", loss.item())


        if total_rmse<total_rmse_min:
            total_rmse_min = total_rmse
            if epoch > 1000:
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_tick.append(elapsed_time)
            #recon_result_best = output_denorm.squeeze(0).cpu().numpy()
            print("--------epoch:",epoch,"loss:",loss.item(),";new minimum rmse:",total_rmse_min)

    return total_rmse_min






if __name__=="__main__":
    #hyperparameters
    nmod = 3


    num_epochs= 5000
    num_epochs_tnn= 5000
    learning_rate1=4e-3
    learning_rate2=4e-3
    learning_rate_tnn= 4e-3

    cuda_is_available=torch.cuda.is_available()
    device=torch.cuda.current_device()
    print("torch.cuda.is_available:",cuda_is_available)
    set_random_seed(231)
    model_dir="./ckpt"
    # # Set the logger
    # utils.set_logger(os.path.join(model_dir, 'train.log'))
    # # Define the model and optimizer


    dropout_rate = 0.3
    model1 = net.FieldFormer_TAP(dropout_rate).cuda()
    model2 = net.FieldFormer_MHTAP(dropout_rate).cuda()

    tnn_model = net.TNN().cuda() if cuda_is_available else net.TNN()

    optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate1)
    optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate2)

    optimizer_tnn = optim.AdamW(tnn_model.parameters(), lr=learning_rate_tnn)
    total_params1 = sum(p.numel() for p in model1.parameters())
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Total number of parameters of TAPNet_sparse: {total_params1}, TAPNet_multi_sparse:{total_params2}")
    # fetch loss function and metrics
    #loss_fn = net.loss_fn_mse
    loss_fn = net.loss_fn_mse


    tap_rmse_min = train_and_evaluate(model1,  optimizer1, loss_fn, x_ob_norm, x_true, observation_tensor, TV=False)
    #mhtap_rmse_min = train_and_evaluate(model2, optimizer2, loss_fn, x_ob_norm, x_true, observation_tensor,  TV=False)
    #tnn_rmse_min = train_and_evaluate_tnn(tnn_model, optimizer_tnn, loss_fn, x_ob_norm, x_true, observation_tensor, TV=False)
    #print("Oberservation rate:",sampling_rate, " TAP RMSE:",tap_rmse_min," MHTAP RMSE:", mhtap_rmse_min, " TNN RMSE:", tnn_rmse_min)
