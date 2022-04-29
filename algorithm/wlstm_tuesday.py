import scipy.io as scio
import random
import torch
from torch import nn,optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
#
from torchensemble import VotingRegressor
from torchensemble.gradient_boosting import GradientBoostingRegressor
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io

from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
from tqdm import tqdm,trange

#
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
#
class Mylstm ( nn.Module ) :
    def __init__ ( self , input_size , hidden_size , num_layers , output_size ) :
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

    def forward ( self , x ) :
        x = x.view ( -1 , x.size ( 0 ) , self.input_size )
        # x=torch.tanh(x)
        H_out , (_ , _) = self.lstm ( x , None )

        Out = self.dense ( H_out )

        return Out.view(-1)
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
#
def R2(obs,pred):
    up=np.sum(np.square(pred-obs))
    down=np.sum(np.square(np.average(obs)-obs))
    return (1-up/down)
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
    calibration=data.iloc[0:round(len(data)*0.666),:].reset_index()
    optim_keeper=pd.DataFrame()
    for level in data.columns:
        if level != 'sum':
            inter_results=pd.DataFrame(index=[np.arange(0,500)],
                                 columns=['train_mse','test_mse',
                                          'num_layers','hidden_size','batch_size','learning_rate','weight_decay','level','Epoch'])
            clib_X,clib_Y=sliding_windows(calibration[level],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
            #
            train_X, test_X, train_Y, test_Y= train_test_split (clib_X,clib_Y, test_size=0.3,random_state=40)
            train_X, test_X,train_Y, test_Y = torch.tensor((train_X), dtype=torch.float).to(device),\
                                              torch.tensor((test_X), dtype=torch.float).to(device), \
                                              torch.tensor((train_Y), dtype=torch.float).to(device),\
                                              torch.tensor((test_Y), dtype=torch.float).to(device)

            i=0
            for Epoch,num_layers,hidden_size,batch_size,learning_rate,weight_decay in product(*Hyper_params):
                if i <=input_parameter['M']:
                    train_loader = Data.DataLoader(Data.TensorDataset(train_X, train_Y), batch_size=batch_size,
                                               shuffle=True, num_workers=0)

                    test_loader = Data.DataLoader(Data.TensorDataset(test_X, test_Y), batch_size=batch_size,
                                              shuffle=True, num_workers=0)

                    model = VotingRegressor(
                        estimator=Mylstm(input_size = input_parameter['input_size'] ,
                            hidden_size = hidden_size ,
                            num_layers = num_layers,
                            output_size = input_parameter['output_size'] ,
                            ),
                        n_estimators=input_parameter['num'],
                        cuda=use_cuda
                    )
                    model.set_optimizer('Adam',  # parameter optimizer
                                        lr=learning_rate,  # learning rate of the optimizer
                                        weight_decay=weight_decay)  # weight decay of the optimizer
                    # Training
                    model.fit(train_loader=train_loader,  # training data
                              epochs=Epoch,  # the number of training epochs
                              test_loader=test_loader,
                              save_model=None)
                    #
                    train_pred = model.predict(train_X).to(device)
                    test_pred = model.predict(test_X).to(device)
                    inter_results.iloc[i, 0], inter_results.iloc[i, 1] = F.mse_loss(train_Y, train_pred).cpu().detach().numpy(),F.mse_loss(test_Y, test_pred).cpu().detach().numpy()
                    inter_results.iloc[i, 2], inter_results.iloc[i, 3], inter_results.iloc[i, 4], inter_results.iloc[i, 5], inter_results.iloc[i, 6] = num_layers,hidden_size,batch_size,learning_rate,weight_decay
                    inter_results.iloc[i, 7],inter_results.iloc[i, 8]=level,Epoch
                    i+=1
            optims = inter_results.dropna().sort_values('test_mse',ascending=True).reset_index(drop=True).iloc[0:1]
            optims['num_of_model']="0 to 14"
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
    calibration=data.iloc[0:round(len(data)*0.666),:].reset_index()
    prediction=data.iloc[round(len(data)*0.666):len(data),:].reset_index()
    optim_keeper = optim_keeper.loc[lambda x: x['time_lag'] == input_parameter['time_lag']]
    #
    final_results = pd.DataFrame(index=[np.arange(0,10)],
                                 columns=['pred_r2', 'pred_rmse', 'pred_nrmse',
                                          'clib_r2','clib_rmse','clib_nrmse'])

    pred_pred,clib_pred = 0,0
    pred_y,clib_y=0,0
    i=0
    for level in optim_keeper['level']:
        level_df= optim_keeper.loc[lambda x: x['level'] == level]
        Epoch,num_layers,hidden_size,batch_size,learning_rate,weight_decay=level_df['Epoch'].item(),\
                                                                           level_df['num_layers'].item(),\
                                                                             level_df['hidden_size'].item(),\
                                                                             level_df['batch_size'].item(),\
                                                                             level_df['learning_rate'].item(),\
                                                                             level_df['weight_decay'].item()

        clib_X,clib_Y=sliding_windows(calibration[level],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        pred_X,pred_Y=sliding_windows(prediction[level],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        #
        train_X, test_X, train_Y, test_Y= train_test_split (clib_X,clib_Y, test_size=0.3,random_state=40)
        train_X, test_X,train_Y, test_Y = torch.tensor((train_X), dtype=torch.float).to(device), \
                                                         torch.tensor((test_X), dtype=torch.float).to(device), \
                                                         torch.tensor((train_Y), dtype=torch.float).to(device),\
                                                         torch.tensor((test_Y), dtype=torch.float).to(device)
        clib_X, pred_X= torch.tensor((clib_X), dtype=torch.float).to(device),\
                        torch.tensor((pred_X), dtype=torch.float).to(device)

        train_loader = Data.DataLoader(Data.TensorDataset(train_X, train_Y), batch_size=batch_size,
                                               shuffle=True, num_workers=0)
        test_loader = Data.DataLoader(Data.TensorDataset(test_X, test_Y), batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        #
        model = VotingRegressor(
            estimator=Mylstm(input_size = input_parameter['input_size'] ,
                hidden_size = hidden_size ,
                num_layers = num_layers,
                output_size = input_parameter['output_size'] ,
                ),
            n_estimators=input_parameter['num'],
            cuda=use_cuda
        )
        model.set_optimizer('Adam',  # parameter optimizer
                            lr=learning_rate,  # learning rate of the optimizer
                            weight_decay=weight_decay)  # weight decay of the optimizer
        # Training
        model.fit(train_loader=train_loader,  # training data
                  epochs=Epoch,  # the number of training epochs
                  test_loader=test_loader,
                  save_model=None)

        pred_y += pred_Y
        clib_y += clib_Y
        pred_pred += model.predict(pred_X).cpu().numpy()
        clib_pred += model.predict(clib_X).cpu().numpy()

    final_results.iloc[i, 0], final_results.iloc[i, 1],final_results.iloc[i, 2]= R2(pred_y, pred_pred), RMSE(pred_y, pred_pred), NRMSE(pred_y, pred_pred)

    final_results.iloc[i, 3], final_results.iloc[i, 4],final_results.iloc[i, 5]= R2(clib_y, clib_pred), RMSE(clib_y, clib_pred), NRMSE(clib_y, clib_pred)
    return final_results





if __name__ == '__main__':
    ##---------------- prepare model input and data-------------- ##
    input_parameter=dict(
        myseed=1, # reproducible
        input_size=3,
        output_size=1,
        time_lag=2, # 0, 1, and 2; representing 1,2 and 3 time lag in output
        num=15,# ensembles
        M=500, # tuning number restriction
        no_cuda=False,
        resolution="day"
    )
    # data prepare
    base=scio.loadmat(f"~/data/dwt_data/{input_parameter['resolution']}_tuesday.mat")
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
    optims = pd.read_excel("~/optimal_hyperparameter/wlstm_tuesday_optimize_params.xlsx",dtype=object)
    final_results=model_prediction(input_parameter=input_parameter,optim_keeper=optims,data=data)
