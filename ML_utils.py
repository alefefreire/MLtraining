from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,fbeta_score
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

class Cross_valid_clf():
  """ 
    This class does cross validation for general classifiers. 
        model: Sklearn model or customized model with fit and predict methods;
        X: features
        y:target
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
  def rmse(self, reg,verbose=True):
    #rmse
        rmse = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            reg.fit(self.X[tr],self.y[tr])
            y_pred=reg.predict(self.X[te])
            rmse.append(mean_squared_error(y_pred,self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'RMSE: {rmse[i]}')
            i+=1
        if verbose:
            print(f'\n Overall RMSE:')
            print(f'RMSE Mean: {np.mean(rmse)}   Std: {np.std(rmse)}')
        return np.mean(rmse)
    #mae
  def mae(self, reg,verbose=True):
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
        return np.mean(mae)
  def r2(self, reg,verbose=True):
        mae = []
        i=0
        for tr, te in self.kf.split(self.X,self.y):
            reg.fit(self.X[tr],self.y[tr])
            y_pred=reg.predict(self.X[te])
            mae.append(r2_score(y_pred,self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'R2: {mae[i]}')
            i+=1
        if verbose:
            print(f'\n Overall R2:')
            print(f'R2 Mean: {np.mean(mae)}   Std: {np.std(mae)}')
        return np.mean(mae)  
    #precision score
