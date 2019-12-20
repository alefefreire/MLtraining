import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *

def cross_valid(model,x,folds,metric,stratified=True,beta=0.6,verbose=True):
    """ 
    This function does cross validation for general classifiers. 
        model: Sklearn model or customized model with fit and predict methods;
        x : Data as a numpy matrix containg with ***the last column as target***;
        folds: Number of folds;
        metrics : 'acc': accuracy,'f1score','prec': precision,'rec':recall,'fbeta': fbeta score;
        stratified: Use stratified Kfold to keep the ratio of classes in all folds;
        beta: Beta parameter for fbeta score metric;
        verbose: Flag to print report over iterations;
        
    returns: List with scores over the folders
    """    
        
    score=[]

    if stratified: 
        kf = StratifiedKFold(folds,shuffle=False,random_state=0)
        i=0
        for train_index, test_index in kf.split(x[:,:-1],x[:,-1]):
            xtrain = x[train_index,:]
            xtest = x[test_index,:]
            model.fit(xtrain[:,:-1],xtrain[:,-1])

            ypred = model.predict(xtest[:,:-1])
            if metric == 'acc':
                score.append(accuracy_score(xtest[:,-1],ypred))
            elif metric == 'f1score':
                score.append(f1_score(xtest[:,-1],ypred))
            elif metric == 'prec':    
                score.append(precision_score(xtest[:,-1],ypred))
            elif metric == 'rec':
                score.append(recall_score(xtest[:,-1],ypred))
            else:
                score.append(fbeta_score(xtest[:,-1],ypred,beta))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {folds}')
                print(f'{metric}: {score[i]}')

            i+=1
        
    else:          
        kf = KFold(folds,shuffle=False,random_state=0) 

        i=0
        for train_index, test_index in kf.split(x):
            xtrain = x[train_index,:]
            xtest = x[test_index,:]
            model.fit(xtrain[:,:-1],xtrain[:,-1])
            ypred = model.predict(xtest[:,:-1])

            if metric == 'acc':
                score.append(accuracy_score(xtest[:,-1],ypred))
            elif  metric == 'f1score':
                score.append(f1_score(xtest[:,-1],ypred))
            elif metric == 'prec':    
                score.append(precision_score(xtest[:,-1],ypred))
            elif metric == 'rec':
                score.append(recall_score(xtest[:,-1],ypred))
            else:
                score.append(fbeta_score(xtest[:,-1],ypred,beta))
            if verbose:
                print('-'*30)
                print(f'\nFold {i+1} out of {folds}')
                print(f'{metric}: {score[i]}')

            i+=1
    if verbose:
        print(f'\n Overall Score:')
        print(f'{metric}:    Mean: {np.mean(score)}   Std: {np.std(score)}')

    return score