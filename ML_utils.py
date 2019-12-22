from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
import numpy as np

class Cross_valid_clf():
  def __init__(self, X, y,n_splits):
    self.n_splits = n_splits
    self.X = X
    self.y = y
  #score method
  def score(self, clf,stratified=True,verbose=True):
    score = []
    if stratified:
        self.kf = StratifiedKFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(X)
        i=0
        for tr, te in self.kf.split(X):
            clf.fit(self.X[tr],self.y[tr])
            score.append(clf.accuracy_score(self.X[te],self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'Accuracy_score: {score[i]}')
            i+=1
    else:
        self.kf = KFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(X)
        for tr, te in self.kf.split(X):
            clf.fit(self.X[tr],self.y[tr])
            score += clf.score(self.X[te],self.y[te])
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
  
  def f1score(self, clf,stratified=True,verbose=True):
    f1score = []
    if stratified:
        self.kf = StratifiedKFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(X)
        i=0
        for tr, te in self.kf.split(X):
            clf.fit(self.X[tr],self.y[tr])
            f1score.append(clf.accuracy_score(self.X[te],self.y[te]))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'f1_score: {f1score[i]}')
            i+=1
    else:
        self.kf = KFold(self.n_splits,shuffle=False,random_state=0)
        self.kf.get_n_splits(X)
        for tr, te in self.kf.split(X):
            clf.fit(self.X[tr],self.y[tr])
            score += clf.score(self.X[te],self.y[te])
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {self.n_splits}')
                print(f'f1_score: {f1score[i]}')
            i+=1
    if verbose:
        print(f'\n Overall Score:')
        print(f'f1_score Mean: {np.mean(f1score)}   Std: {np.std(f1score)}')
    return np.mean(f1score)