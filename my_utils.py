import pandas as pd
import numpy as np
from scipy.stats import mode, gaussian_kde
from scipy.optimize import minimize, shgo
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=2)
from sklearn.model_selection import GridSearchCV




def rrmse(y_true,y_pred):
    return np.sqrt(mse(y_true,y_pred))/np.mean(y_true)

def split_df(data,ids,val_size,id_col,shuffle=True):
    if shuffle: rn.shuffle(camps)

    length = len(ids)

    split = int(val_size * length)

    idx_val = ids[-split:]
    idx_tr = ids[:-split]
    
    return data[data[id_col].isin(idx_tr)],data[data[id_col].isin(idx_val)]

def cross_valid(model,x,folds,metric,verbose=True):
    """ 
    This function does cross validation for general regressors. 
        model: Sklearn model or customized model with fit and predict methods;
        x : Data as a numpy matrix containg with ***the last column as target***;
        folds: Number of folds;
        metrics : 'mae': mse,'rmse','rrmse'
        verbose: Flag to print report over iterations;
        
    returns: List with scores over the folders
    """    

    score=[]
    

    kf = KFold(folds,shuffle=False,random_state=0) 


    i=0
    for train_index, test_index in kf.split(x):

        xtrain = x[train_index,:]
        xtest = x[test_index,:]

        model.fit(xtrain[:,:-1],xtrain[:,-1])

        ypred = model.predict(xtest[:,:-1])

        ytrue= xtest[:,-1] 
          
              
        if metric == 'mae':
            score.append(mae(ytrue,ypred))
        elif metric == 'mse':
            score.append(mse(ytrue,ypred))
        elif metric == 'rrmse':
            score.append(rrmse(ytrue,ypred))

        else:
            score.append(rmse(xtest[:,-1],ypred))

        if verbose:
            print('-'*30)
            print(f'\nFold {i+1} out of {folds}')
            print(f'{metric}: {score[i]}')

        i+=1

    if verbose:
        print(f'\n Overall Score:')
        print(f'{metric}:    Mean: {np.mean(score)}   Std: {np.std(score)}')


    return score
def cross_valid_key(model,x,key,preds,target,metric,verbose=True):
    """ 
    This function does cross validation for general regressors. 
        model: Sklearn model or customized model with fit and predict methods;
        x : Data as a numpy matrix containg with ***the last column as target***;
        key: Column name containing keys for spliting the folds;
        metrics : 'mae': mse,'rmse','rrmse'
        verbose: Flag to print report over iterations;
        
    returns: List with scores over the folders
    """    

    score=[]
    
    keys = x[key].unique().tolist()
 


    for idx, item in enumerate([1,2,3,4,5]):

        xtrain,xtest = split_camp(x,keys,0.2)
        
        model.fit(xtrain[feat],xtrain[target])

        ypred = model.predict(xtest[feat])
        
        ytrue= xtest[target].values 
          
        if metric == 'mae':
            score.append(mae(ytrue,ypred))
        elif metric == 'mse':
            score.append(mse(ytrue,ypred))
        elif metric == 'rrmse':
            score.append(rrmse(ytrue,ypred))

        else:
            score.append(rmse(xtest[target].tolist(),ypred))

        if verbose:
            print('-'*30)
            print(f'\nFold {idx} out of 5')
            print(f'Key {item}')
            print(f'{metric}: {score[idx]}')

 

    if verbose:
        print(f'\n Overall Score:')
        print(f'{metric}:    Mean: {np.mean(score)}   Std: {np.std(score)}')


    return score

def kde(array, cut_down=True, bw_method='scott'):
    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        f_mean = counts.mean()
        f_above_mean = bins[counts > f_mean]
        bounds = [f_above_mean.min(), f_above_mean.max()]
        array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]
    return gaussian_kde(array, bw_method=bw_method)

def mode_estimation(array, cut_down=True, bw_method='scott'):
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    bounds = np.array([[array.min(), array.max()]])
    results = shgo(lambda x: -kernel(x)[0], bounds=bounds, n=100*len(array))
    return results.x[0]

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    with plt.style.context(style):    
        xticks = np.arange(0,lags)
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return

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
 
        #print("Eliminando variáveis: {}".format(labels))
        series.drop(labels,axis=1,inplace=True)  
 
    return series

class Cross_valid_clf():
  """ 
    This class does cross validation for general classifiers. 
        model: Sklearn model or customized model with fit and predict methods;
        X: array with values for features
        y:array with values for target
        folds: Number of folds;
        metrics : accuracy,f1score, precision,recall,fbeta score;
        stratified: Use stratified Kfold to keep the ratio of classes in all folds;
        beta: Beta parameter for fbeta score metric;
        verbose: Flag to print report over iterations;
        
    returns: List with scores over the folders
  """            
  def __init__(self, X, y,n_splits,stratified=True):
    self.n_splits = n_splits
    self.X = X
    self.y = y
        
    if stratified:
        self.kf=StratifiedKFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(self.X)
    else:
        self.kf=KFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(self.X)

  #score method
  def score(self, clf,verbose=True):
        score = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            clf.fit(self.X[tr],self.y[tr])
            score.append(clf.score(self.X[te],self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'Accuracy_score: {score[i]}')
            i+=1
        if verbose:
            print(f'\n Overall Score:')
            print(f'Accuracy_score Mean: {np.mean(score)}   Std: {np.std(score)}')
        return np.mean(score)
    #f1score method
  def f1score(self, clf,verbose=True):
        f1score = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            clf.fit(self.X[tr],self.y[tr])
            y_pred=clf.predict(self.X[te])
            f1score.append(f1_score(y_pred,self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'f1_score: {f1score[i]}')
            i+=1
        if verbose:
            print(f'\n Overall f1score:')
            print(f'f1score Mean: {np.mean(f1score)}   Std: {np.std(f1score)}')
        return np.mean(f1score)
    #precision score
  def precisionscore(self, clf,verbose=True):
        prec_score = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            clf.fit(self.X[tr],self.y[tr])
            y_pred=clf.predict(self.X[te])
            prec_score.append(precision_score(y_pred,self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'Precision_score: {prec_score[i]}')
            i+=1
        if verbose:
            print(f'\n Overall Score:')
            print(f'Precision_score Mean: {np.mean(prec_score)}   Std: {np.std(prec_score)}')
        return np.mean(prec_score)
    #Recall score      
  def recallscore(self, clf,verbose=True):
        rec_score = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            clf.fit(self.X[tr],self.y[tr])
            y_pred=clf.predict(y_pred,self.X[te])
            rec_score.append(recall_score(self.X[te],self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'Recall_score: {rec_score[i]}')
            i+=1
        if verbose:
            print(f'\n Overall Score:')
            print(f'Recall_score Mean: {np.mean(rec_score)}   Std: {np.std(rec_score)}')
        return np.mean(rec_score)
    #fbeta score
  def fbetascore(self, clf,verbose=True,beta=0.6):
        fbetascore = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            clf.fit(self.X[tr],self.y[tr])
            y_pred=clf.predict(self.X[te])
            fbetascore.append(fbeta_score(y_pred,self.y[te],beta))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'fbeta_score: {fbetascore[i]}')
            i+=1
        if verbose:
            print(f'\n Overall Score:')
            print(f'fbeta_score Mean: {np.mean(fbetascore)}   Std: {np.std(fbetascore)}')
        return np.mean(fbetascore)

class Cross_valid_reg():
  """ 
    This class does cross validation for general regressors. 
        model: Sklearn model or customized model with fit and predict methods;
        x : features;
        y: target
        folds: Number of folds;
        metrics : RMSE =root mean squared error; MAE= mean absolute error
        stratified: Use stratified Kfold to keep the ratio of classes in all folds;
        verbose: Flag to print report over iterations;
        
    returns: List with scores over the folders
  """    
  def __init__(self, X, y,n_splits,stratified=True):
    self.n_splits = n_splits
    self.X = X
    self.y = y
        
    if stratified:
        self.kf=StratifiedKFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(self.X)
    else:
        self.kf=KFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(self.X)

  #score method
  def rmse(self, reg,verbose=True,overall=True):
    #rmse
        rmse = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            reg.fit(self.X[tr],self.y[tr])
            y_pred=reg.predict(self.X[te])
            rmse.append(np.sqrt(mean_squared_error(y_pred,self.y[te])))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'RMSE: {rmse[i]}')
            i+=1
        if verbose:
            print(f'\n Overall RMSE:')
            print(f'RMSE Mean: {np.mean(rmse)}   Std: {np.std(rmse)}')
        if overall:
            return np.mean(rmse)
        else:
            return rmse
    #mae
  def mae(self, reg,verbose=True,overall=True):
        mae = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            reg.fit(self.X[tr],self.y[tr])
            y_pred=reg.predict(self.X[te])
            mae.append(mean_absolute_error(y_pred,self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'MAE: {mae[i]}')
            i+=1
        if verbose:
            print(f'\n Overall MAE:')
            print(f'MAE Mean: {np.mean(mae)}   Std: {np.std(mae)}')
        if overall:
            return np.mean(mae)
        else:
            return mae
  def r2(self, reg,verbose=True,overall=True):
        r2 = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            reg.fit(self.X[tr],self.y[tr])
            y_pred=reg.predict(self.X[te])
            mae.append(r2_score(y_pred,self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'R2: {r2[i]}')
            i+=1
        if verbose:
            print(f'\n Overall R2:')
            print(f'R2 Mean: {np.mean(r2)}   Std: {np.std(r2)}')
        if overall:
            return np.mean(r2)
        else:
            return r2  
    #precision score

def feature_importance_plot(algorithm,X_train,y_train,of_type):
    """This function does the feature importance for any classifiers or regressors.

    Parameters
    ----------------
    algorithm: Algorithm which one wants to importance the relevant features
    X_train: axis x of the train dataframe
    y_train: axis y of the target dataframe
    of_type: 'coef' or 'feat', depending on the algorithm.

    Return
    -----------------
    Plot with feature importances

    """
    if of_type == "coef":
        algorithm.fit(X_train,y_train)
        coef = pd.DataFrame(algorithm.coef_.ravel())
        coef["coef"] = X_train.columns
        plt.figure(figsize=(14,4))
        ax1 = sns.barplot(coef["coef"],coef[0],palette="jet_r",
                          linewidth=2,edgecolor="k"*coef["coef"].nunique())
        #ax1.set_facecolor("lightgrey")
        ax1.axhline(0,color="k",linewidth=2)
        plt.ylabel("coefficients")
        plt.xlabel("features")
        plt.xticks(rotation='vertical')
        plt.title('FEATURE IMPORTANCES')
    
    elif of_type == "feat":
        algorithm.fit(X_train,y_train)
        coef = pd.DataFrame(algorithm.feature_importances_)
        coef["feat"] = X_train.columns
        plt.figure(figsize=(14,4))
        ax2 = sns.barplot(coef["feat"],coef[0],palette="jet_r",
                          linewidth=2,edgecolor="k"*coef["feat"].nunique())
        #ax2.set_facecolor("lightgrey")
        ax2.axhline(0,color="k",linewidth=2)
        plt.ylabel("coefficients")
        plt.xlabel("features")
        plt.xticks(rotation='vertical')
        plt.title('FEATURE IMPORTANCES')
def algorithm_grid_search_cv(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred
