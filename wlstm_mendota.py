import scipy.io as scio
import random
import torch
from torch import nn,optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adam
from itertools import product
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import math
from tqdm import tqdm,trange
import time
#
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
#
class Mylstm ( nn.Module ) :
    def __init__ ( self , input_size , hidden_size , num_layers , output_size) :
        super ( Mylstm , self ).__init__ ( )  # inherit nn.Module's attributions

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.output_size = output_size
        ##define layers
        self.lstm = nn.LSTM (
            input_size = self.input_size ,  # The number of features in the input x
            hidden_size = self.hidden_size ,  # The number of features in the hidden state h
            num_layers = self.num_layers ,  # The Number of recurrent layers
            batch_first = True
        )
        # define output layers
        self.dense = nn.Linear (
            in_features = self.hidden_size ,  # size of each input sample
            out_features = self.output_size  # size of each output sample
        )
        # self.dense = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size*2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p = 0.2),
        #     nn.Linear(self.hidden_size*2, self.output_size)
        #     )
    def forward ( self , x ,h_state=None) :
        H_out , h_state = self.lstm ( x , h_state )
        Out = self.dense ( H_out )
        return Out.view(-1),h_state
#
def sliding_windows(data, seq_length,time_lag):
    x = []
    y = []
    for i in range(len(data)-seq_length-1-time_lag):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length+time_lag]
        x.append(_x)
        y.append(_y)
    return np.array(x),np.array(y)
#
def mini_batch_train (model,loss_func, train_loader,optimizer,device,h_state=None ):
    model.train()
    for batch, (train_x, train_y) in enumerate(train_loader):
        # get the inputs -- a list of [inputs, labels]
        train_x, train_y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad()
        train_output,h_state = model(train_x,h_state)
        h_state = (h_state[0].data,h_state[1].data)
        loss = loss_func(train_output, train_y)
        loss.backward(retain_graph=True)
        optimizer.step()

def get_all_preds(model,X_loader,device):
    with torch.no_grad ( ) :
        all_preds=torch.tensor([]).float().to(device)
        for _,features in enumerate(X_loader):
            preds,_ =model(features[0])
            all_preds=torch.cat((all_preds,preds),dim = 0)
        return all_preds
#
def R2(obs,pred):
    up=np.sum(np.square(pred-obs))
    down=np.sum(np.square(np.average(obs)-obs))
    return ((1-up/down+eps))
def RMSE(obs,pred):
    return sqrt(mean_squared_error(obs,pred))
def NRMSE(obs,pred):
    return RMSE(obs,pred)/(max(obs)-min(obs))
#

def param_optimizer(input_parameter,Hyper_params,data):
    # reproducible
    random.seed(input_parameter['myseed'])
    torch.manual_seed(input_parameter['myseed'])
    np.random.seed(input_parameter['myseed'])
    g = torch.Generator()
    g.manual_seed(input_parameter['myseed'])
    #
    use_cuda = not input_parameter['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #
    calibration=data.iloc[0:round(len(data)*0.8),:].reset_index()
    optim_keeper=pd.DataFrame()
    #
    for level in data.columns:
        if level == 'sum': break
        print("level:",level)
        clib_X,clib_Y=sliding_windows(calibration[level],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        #
        for k in trange(input_parameter['num']):
            train_X, test_X, train_Y, test_Y= train_test_split (clib_X, clib_Y, test_size=0.3,random_state=k)

            train_X, train_Y = torch.as_tensor(train_X, dtype=torch.float).to(device),\
                               torch.as_tensor(train_Y, dtype=torch.float).to(device)
            test_X,test_Y = torch.as_tensor(test_X, dtype=torch.float).to(device),\
                            torch.as_tensor(test_Y, dtype=torch.float).to(device)
            inter_results=pd.DataFrame(index=range(0,9999),
                                       columns=['train_mse','test_mse','level',
                                                'Epoch','num_layers','hidden_size','batch_size','learning_rate','weight_decay'])
            i=0
            for Epoch,num_layers,hidden_size,batch_size,learning_rate,weight_decay in product(*Hyper_params): ## ensemble training loop

                train_loader = Data.DataLoader(Data.TensorDataset(train_X, train_Y), batch_size=batch_size,
                                           shuffle=True, num_workers=0,worker_init_fn=seed_worker,generator=g)

                model = Mylstm(input_size = input_parameter['input_size'] ,
                        hidden_size = hidden_size ,
                        num_layers = num_layers ,
                        output_size = input_parameter['output_size']
                        )
                optimizer = optim.Adam(model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
                loss_func = nn.MSELoss()
                model = model.to(device)
                torch.cuda.empty_cache()
                for epoch in range(Epoch):
                    mini_batch_train(model,loss_func, train_loader,optimizer,device,h_state=None)
                #train and test recording
                train_X_loader=Data.DataLoader(Data.TensorDataset(train_X), batch_size=batch_size,
                                    shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=g)
                test_X_loader=Data.DataLoader(Data.TensorDataset(test_X), batch_size=batch_size,
                                    shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=g)
                train_pred= get_all_preds(model,train_X_loader,device)
                test_pred= get_all_preds(model,test_X_loader,device)
                inter_results.iloc[i, 0], inter_results.iloc[i, 1] =\
                    F.mse_loss(train_Y, train_pred).cpu().detach().numpy(),\
                    F.mse_loss(test_Y, test_pred).cpu().detach().numpy()
                inter_results.iloc[i, 2]= level
                inter_results.iloc[i, 3], inter_results.iloc[i, 4], inter_results.iloc[i, 5], inter_results.iloc[i, 6], inter_results.iloc[i, 7],inter_results.iloc[i, 8] =\
                    Epoch,num_layers,hidden_size,batch_size,learning_rate,weight_decay
                i+=1
                if i%25==0: print(i,'in',input_parameter['M'])
                if i == input_parameter['M']: break
            optims = inter_results.dropna().sort_values('test_mse',ascending=True).reset_index(drop=True).iloc[0:1]
            optims['num_of_model']=k
            optim_keeper=pd.concat((optim_keeper,optims)).dropna()
    return optim_keeper

def model_prediction(input_parameter=None,optim_keeper=None,data=None):
    # reproducible
    random.seed(input_parameter['myseed'])
    torch.manual_seed(input_parameter['myseed'])
    np.random.seed(input_parameter['myseed'])
    g = torch.Generator()
    g.manual_seed(input_parameter['myseed'])
    #
    use_cuda = not input_parameter['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #
    calibration=data.iloc[0:round(len(data)*0.8),:].reset_index()
    prediction=data.iloc[round(len(data)*0.8):len(data),:].reset_index()
    optim_keeper = optim_keeper.loc[lambda x: x['resolution'] == input_parameter['resolution']]
    #
    final_results = pd.DataFrame(index=[np.arange(0,10)],
                             columns=['pred_r2', 'pred_rmse', 'pred_nrmse',
                                      'clib_r2','clib_rmse','clib_nrmse']
                             )
    pred_pred_final,clib_pred_final = 0,0
    clib_Y_final,pred_Y_final = 0,0
    for level in optim_keeper['level'].drop_duplicates():
        print(level)
        clib_X,clib_Y=sliding_windows(calibration[level],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        pred_X,pred_Y=sliding_windows(prediction[level],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])

        clib_X, pred_X= torch.as_tensor(clib_X, dtype=torch.float).to(device),\
                        torch.as_tensor(pred_X, dtype=torch.float).to(device)
        #
        k_clibs = pd.DataFrame()
        k_preds = pd.DataFrame()
        for k in trange(input_parameter['num']):
            level_df= optim_keeper.loc[lambda x: x['level'] == level].reset_index(drop=True)
            level_df= level_df.loc[k]
            Epoch,num_layers,hidden_size,batch_size,learning_rate,weight_decay=level_df['Epoch'],\
                                                                       level_df['num_layers'],\
                                                                       level_df['hidden_size'],\
                                                                       level_df['batch_size'],\
                                                                       level_df['learning_rate'],\
                                                                       level_df['weight_decay']
            train_X,_, train_Y,_ = train_test_split (clib_X, clib_Y, test_size=0.3,random_state=k)
            train_X, train_Y = torch.as_tensor(train_X, dtype=torch.float).to(device),\
                               torch.as_tensor(train_Y, dtype=torch.float).to(device)

            train_loader = Data.DataLoader(Data.TensorDataset(train_X, train_Y), batch_size=batch_size,
                                       shuffle=True, num_workers=0,worker_init_fn=seed_worker,generator=g)

            model = Mylstm(input_size = input_parameter['input_size'] ,
                    hidden_size = hidden_size ,
                    num_layers = num_layers ,
                    output_size = input_parameter['output_size']
                    )

            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay
                                    )
            loss_func = nn.MSELoss()
            model = model.to(device)
            torch.cuda.empty_cache()
            for epoch in range(Epoch):
                mini_batch_train(model,loss_func, train_loader,optimizer,device,h_state=None)
            #train and test recording
            clib_X_loader=Data.DataLoader(Data.TensorDataset(clib_X), batch_size=batch_size,
                                    shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=g)
            pred_X_loader=Data.DataLoader(Data.TensorDataset(pred_X), batch_size=batch_size,
                                    shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=g)
            clib_pred= get_all_preds(model,clib_X_loader,device)
            pred_pred= get_all_preds(model,pred_X_loader,device)
            clib_pred,pred_pred =clib_pred.cpu().detach().numpy(),pred_pred.cpu().detach().numpy()
            k_clibs = pd.concat((k_clibs, pd.Series(clib_pred)), axis=1)
            k_preds = pd.concat((k_preds, pd.Series(pred_pred)), axis=1)

        clib_Y_final += clib_Y
        pred_Y_final += pred_Y

        clib_pred_final += k_clibs.mean(axis=1)
        pred_pred_final += k_preds.mean(axis=1)

    i=0
    final_results.iloc[i, 0], final_results.iloc[i, 1],final_results.iloc[i, 2] = r2_score(pred_Y_final, pred_pred_final), \
                                                                                   RMSE(pred_Y_final, pred_pred_final), \
                                                                                   NRMSE(pred_Y_final, pred_pred_final)
    final_results.iloc[i, 3], final_results.iloc[i, 4],final_results.iloc[i, 5]= r2_score(clib_Y_final, clib_pred_final),\
                                                                                 RMSE(clib_Y_final, clib_pred_final), \
                                                                                 NRMSE(clib_Y_final, clib_pred_final)
    final_results=final_results.dropna()

    return final_results

if __name__ == '__main__':
    ##---------------- prepare model input and data-------------- ##
    input_parameter=dict(
        myseed=0, # reproducible
        input_size=3,
        output_size=1,
        time_lag=0, # 0, 1, and 2; representing 1,2 and 3 time lag in output
        num=15,# ensembles
        M=500, # tuning number restriction
        no_cuda=False,
        resolution='hour'  # day, hour, month
    )
    # prepare data
    base=scio.loadmat(f"~/data/dwt_data/{input_parameter['resolution']}_mendota.mat")
    data=pd.DataFrame()
    data['ca3'],data['cd3'],data['cd2'],data['cd1']=base['ca3'].reshape(-1),base['cd3'].reshape(-1),base['cd2'].reshape(-1),base['cd1'].reshape(-1)
    data['sum'] = data.apply(sum,1)
    ##---------------- clabration phase-------------- ##
    ## Hyper_params search space
    Hyper=dict(
        Epoch=[25,50,75,100],
        num_layers=[1,2,5],
        hidden_size=[5,25,50],
        batch_size=[16,32,64],
        learning_rate=[0.1,0.01,0.001],
        weight_decay=[5e-4,5e-2,0]
    )
    Hyper_params=[v for v in Hyper.values()]
    optim_keeper=param_optimizer(input_parameter=input_parameter,Hyper_params=Hyper_params,data=data)

    ##---------------- prediction phase-------------- ##
    optims = pd.read_excel("~/optimal_hyperparameter/wlstm_mendota_optimize_params.xlsx",dtype=object)

    final_results=model_prediction(input_parameter=input_parameter,optim_keeper=optims,data=data)
