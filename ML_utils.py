from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
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