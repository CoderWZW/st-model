from loader.XGBdataloader import XGBLoader
from Glob.glob import p_parse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
#from utils.lossFunction import Loss

from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV
from sklearn.linear_model import Ridge, LinearRegression


def getData(args):
    data = np.load(args.data_path, allow_pickle=True)['data'][:,:,0]
    print(data.shape)
    data = data.reshape(data.shape[0], 307)
    print(data.shape)
    train = data[:-19*288]
    test = data[-19*288:]
    print(train.shape)
    print(test.shape)
    print('data split')
    return train, test

def MAPE(y_true, y_pred):
    idx = (y_true>5).nonzero()
    return np.mean(np.abs(y_true[idx] - y_pred[idx]) / y_true[idx])

def MAE(y_true, y_pred):
    return np.mean(abs(y_pred-y_true))

def RMSE(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)**0.5

def ridgeRegression(args):
    train, test = getData(args)
    print(len(train))
    res = []
    trainX = [[i] for i in range(40*288)]
    testX = [[i] for i in range(40*288,59*288)]
    trainX = np.array(trainX)
    testX = np.array(testX)

    print(len(trainX),len(trainX[0]))
    print(len(testX))
    for i in range(307):
        model = Ridge(alpha=0.001,normalize=True)
        model.fit(trainX, train[:,i])
        pred = model.predict(testX)
        res.append(pred)
    res = np.array(res)
    res = res.transpose((1,0))

    print('Ridge: ')
    print('RMSE: {}'.format(RMSE(y_pred=res, y_true=test)))
    print('MAPE: {}'.format(MAPE(y_pred=res, y_true=test)))
    print('MAE: {}'.format(MAE(y_pred=res, y_true=test)))

def lassoRegression(args):
    train, test = getData(args)
    res = []
    trainX = [[i] for i in range(40*288)]
    testX = [[i] for i in range(40*288,59*288)]
    trainX = np.array(trainX)
    testX = np.array(testX)
    
    for i in range(307):
        model = Lasso(alpha=0.001, normalize=True, precompute=False, warm_start=True, copy_X=False, max_iter=5000)
        model.fit(trainX, train[:,i])
        pred = model.predict(testX)
        res.append(pred)
    res = np.array(res)
    res = res.transpose((1,0))
    print(res.shape)

    print('Lasso: ')
    print('RMSE: {}'.format(RMSE(y_pred=res, y_true=test)))
    print('MAPE: {}'.format(MAPE(y_pred=res, y_true=test)))
    print('MAE: {}'.format(MAE(y_pred=res, y_true=test)))

        


if __name__ == "__main__":
    
    args = p_parse()
    ridgeRegression(args)
    lassoRegression(args)
    XGBoost(args)
    #arima(args)
    
