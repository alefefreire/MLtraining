import pandas as pd 
import numpy as np

# Adpated from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/        
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    var_n= data.columns.tolist()
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))                    
        names += [(var_n[j]+'(t-%d)' % ( i)) for j in range(n_vars)]
 
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(var_n[j]+'(t)') for j in range(n_vars)]
        else:
            names += [(var_n[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
 
    return agg
 
 
def dataTimeSeries(timesteps,df,predictors,target,dropnan,out=2,dropVars=True):
    """ 
    This function transforms a dataframe in a timeseries for surpervised learning.
        timesteps: Number of delays (i.e: timesteps =2 (t),(t-1),(t-2));
        df: Dataframe;
        predictors: List of columns in dataframe as features for the ML algorithm;
        target: Target of the supervised learning;
        dropnan: Flag to drop the NaN values after transforming the 
        out: Number of steps to forecast (i.e: out = 2 (t),(t+1));
        dropVars= Leave only the Target of the last timestep on the resulting dataframe;
    """    
    
    series = series_to_supervised(df[predictors+[target]].copy(),timesteps,out,dropnan=dropnan)
 
    if dropnan==False:
        series.replace(pd.np.nan,0,inplace=True)
    
    # Dropping other variables:
    if dropVars:
        index = list(np.arange(series.shape[1]-2,
                               series.shape[1]-len(predictors)-2,
                               -1))
 
        labels = [item  for idx,item in enumerate(series.columns) 
                  if idx in index]
 
        print("Eliminando variáveis: {}".format(labels))
        series.drop(labels,axis=1,inplace=True)  
 
    return series