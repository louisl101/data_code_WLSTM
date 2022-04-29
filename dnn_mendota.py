# louis1001
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import scipy.stats
from math import sqrt
from tqdm import trange
from joblib import dump, load
import pickle
import itertools
import scipy.io as scio
import random
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
def R2(obs,pred):
    up=np.sum(np.square(pred-obs))
    down=np.sum(np.square(np.average(obs)-obs))
    return ((1-up/down))
def RMSE(obs,pred):
    return sqrt(mean_squared_error(obs,pred))
def NRMSE(obs,pred):
    return RMSE(obs,pred)/(max(obs)-min(obs))
#
def param_optimizer(input_parameter,Hyper_params,data):
    # reproducible
    random.seed(input_parameter['myseed'])
    np.random.seed(input_parameter['myseed'])
    #
    calibration=data.iloc[0:round(len(data)*0.8),:].reset_index()
    #
    normaler=StandardScaler()
    optim_keeper=pd.DataFrame()
    #
    for k in trange(input_parameter['num']):
        print("model:",k)
        clib_X,clib_Y=sliding_windows(calibration['algae'],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        train_X, test_X, train_Y, test_Y= train_test_split (clib_X, clib_Y, test_size=0.3,random_state=k)
        #
        normaler.fit(train_X)
        train_X, test_X= normaler.transform(train_X), normaler.transform(test_X)
        inter_results=pd.DataFrame(index=range(0,99999),
                                   columns=['train_mse','test_mse','level',
                                            'Epoch','neurons','batch_size','learning_rate','weight_decay'])
        #
        j=0
        i=0
        for Epoch,num_layers,batch_size,learning_rate,weight_decay in itertools.product(*Hyper_params):
            for neurons in itertools.combinations_with_replacement(input_parameter['hidden_size'], num_layers):
            #
                model = MLPRegressor(hidden_layer_sizes=neurons,
                                     activation='relu',
                                     batch_size=batch_size,
                                     learning_rate_init=learning_rate,
                                     alpha=weight_decay,
                                     max_iter=Epoch,
                                     random_state=40,
                                     solver='adam',
                                     shuffle=True,
                                     validation_fraction=0.3,
                                     early_stopping=True,
                                     verbose=False
                                     )
            #
                model.fit(train_X, train_Y)
                train_pred = model.predict(train_X)
                test_pred = model.predict(test_X)
                #
                inter_results.iloc[i, 0], inter_results.iloc[i, 1], inter_results.iloc[i, 2] = mean_squared_error(train_Y, train_pred), mean_squared_error(test_Y, test_pred), "None"
                inter_results.iloc[i, 3], inter_results.iloc[i, 4], inter_results.iloc[i, 5],inter_results.iloc[i, 6], inter_results.iloc[i, 7]= Epoch,neurons,batch_size,learning_rate,weight_decay
                i+=1
            j+=1
            if j%25==0: print(j,'in',input_parameter['M'])
            if j == input_parameter['M']: break
        optims = inter_results.dropna().sort_values('test_mse',ascending=True).reset_index(drop=True).iloc[0:1]
        optims['num_of_model']=k
        optim_keeper=pd.concat((optim_keeper,optims)).dropna()
    return optim_keeper

def model_prediction(input_parameter=None,optim_keeper=None,data=None):
    # reproducible
    random.seed(input_parameter['myseed'])
    np.random.seed(input_parameter['myseed'])
    #
    calibration=data.iloc[0:round(len(data)*0.8),:].reset_index()
    prediction=data.iloc[round(len(data)*0.8):len(data),:].reset_index()
    optim_keeper = optim_keeper.loc[lambda x: x['resolution'] == input_parameter['resolution']]
    #
    normaler=StandardScaler()

    final_results = pd.DataFrame(index=[np.arange(0,10)],
                                 columns=['pred_r2', 'pred_rmse', 'pred_nrmse',
                                          'clib_r2','clib_rmse','clib_nrmse'])
    #
    k_clibs = pd.DataFrame()
    k_preds = pd.DataFrame()
    for k in trange(input_parameter['num']):
        level_df= optim_keeper.reset_index(drop=True)
        level_df=level_df.loc[k]
        Epoch,batch_size,neurons,learning_rate,weight_decay=level_df['Epoch'], \
                                                            level_df['batch_size'],\
                                                            level_df['neurons'],\
                                                            level_df['learning_rate'],\
                                                            level_df['weight_decay']
        if neurons.__class__ == tuple:
            neurons=neurons
        else: neurons=eval(neurons)
        #
        clib_X,clib_Y=sliding_windows(calibration['algae'],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        pred_X,pred_Y=sliding_windows(prediction['algae'],seq_length=input_parameter['input_size'],time_lag=input_parameter['time_lag'])
        ## keep the same normaler as training process
        train_X, test_X, train_Y, test_Y= train_test_split (clib_X, clib_Y, test_size=0.3,random_state=k)
        #
        normaler.fit(train_X)
        train_X, test_X= normaler.transform(train_X), normaler.transform(test_X)        #
        clib_X, pred_X= normaler.transform(clib_X), normaler.transform(pred_X)        #
        model = MLPRegressor(hidden_layer_sizes=neurons,
                                 activation='relu',
                                 batch_size=batch_size,
                                 learning_rate_init=learning_rate,
                                 alpha=weight_decay,
                                 max_iter=Epoch,
                                 random_state=40,
                                 solver='adam',
                                 shuffle=True,
                                 validation_fraction=0.3,
                                 early_stopping=True,
                                 verbose=False
                                 )
        #
        model.fit(train_X, train_Y)
        clib_pred = model.predict(clib_X)
        pred_pred = model.predict(pred_X)
        #
        k_clibs = pd.concat((k_clibs, pd.Series(clib_pred)), axis=1)
        k_preds = pd.concat((k_preds, pd.Series(pred_pred)), axis=1)
    clib_Y_final = clib_Y
    pred_Y_final = pred_Y
    clib_pred_final = k_clibs.mean(axis=1)
    pred_pred_final = k_preds.mean(axis=1)
    i=0
    final_results.iloc[i, 0], final_results.iloc[i, 1],final_results.iloc[i, 2] = R2(pred_Y_final, pred_pred_final), \
                                                                                   RMSE(pred_Y_final, pred_pred_final), \
                                                                                   NRMSE(pred_Y_final, pred_pred_final)
    final_results.iloc[i, 3], final_results.iloc[i, 4],final_results.iloc[i, 5]= R2(clib_Y_final, clib_pred_final),\
                                                                                 RMSE(clib_Y_final, clib_pred_final), \
                                                                                 NRMSE(clib_Y_final, clib_pred_final)
    final_results=final_results.dropna().reset_index(drop=True)
    return final_results


if __name__ == '__main__':
    ##---------------- prepare model input and data-------------- ##
    input_parameter=dict(
        myseed=20, # reproducible
        input_size=3,
        output_size=1,
        time_lag=0, # 0, 1, and 2; representing 1,2 and 3 time lag in output
        num=15,# ensembles
        M=500, # tuning number restriction
        hidden_size=[5,25,50,100,200],  #pre-defined hidden units
        resolution='hour'  # day, hour, month
    )
    # data prepare
    data=pd.read_csv(f"~/data/original_data/{input_parameter['resolution']}_mendota.csv")

    ##---------------- clabration phase-------------- ##
    # # Hyper_params search space
    Hyper=dict(
        Epoch=[75],
        num_layers=[1,3,5],
        batch_size=[16,32,64],
        learning_rate=[0.1,0.01,0.001],
        weight_decay=[5e-4,5e-2]
    )
    Hyper_params=[v for v in Hyper.values()]
    optim_keeper=param_optimizer(input_parameter=input_parameter,Hyper_params=Hyper_params,data=data)

    ##---------------- prediction phase-------------- ##
    optims = pd.read_excel("~/optimal_hyperparameter/dnn_mendota_optim_keeper.xlsx",dtype=object)
    final_results=model_prediction(input_parameter=input_parameter,optim_keeper=optims,data=data)
