from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,fbeta_score
import numpy as np

class Cross_valid_clf():
    
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
            f1score.append(f1_score(self.X[te],self.y[te]))
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
            prec_score.append(precision_score(self.X[te],self.y[te]))
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
            fbetascore.append(fbeta_score(self.X[te],self.y[te],beta))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'fbeta_score: {fbetascore[i]}')
            i+=1
        if verbose:
            print(f'\n Overall Score:')
            print(f'fbeta_score Mean: {np.mean(fbetascore)}   Std: {np.std(fbetascore)}')
        return np.mean(fbetascore)  