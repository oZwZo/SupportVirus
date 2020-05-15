from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def tuning_C_given(SV_ls):
    """
    given model list of last training , delve into the peak range and search better performance
    """
    best_C=np.argmax([model.auc for model in SV_ls])
    C_ls = [model.C for model in SV_ls]
    if best_C != 0:
        lower=C_ls[best_C-1]
    else:
        lower=C_ls[best_C]
    if best_C != len(C_ls) -1:
        upper=C_ls[best_C+1]
    else:
        upper=C_ls[best_C]

    return np.linspace(lower,upper,10)

def printting_training_result(SV_ls):
    print('-----------------------')
    print('   C \t \t  AUC')
    print('-----------------------')
    for model in SV_ls:
        print(' %.3f \t \t %.4f'%(model.C,model.auc))
    print('-----------------------')

def SVC_model(X_train,X_val,y_train,y_val,**kwarg):
    """ 
    given Training set and validation set, return a model with y_proba,roc curve,ROC_AUC
    arguments:    
        X_train, np array
        X_val,   np array
        y_train, shape (n,)
        y_val,   shape (n,)
        **kwarg  : the key word arguments to class SVC()
    """
    model = SVC(probability=True,**kwarg)
    model.fit(X_train,y_train)
    model.y_proba=model.predict_proba(X_val)
    model.roc = roc_curve(y_true=y_val,y_score=model.y_proba[:,1])
    model.auc = roc_auc_score(y_true=y_val,y_score=model.y_proba[:,1])
    return model

def AUC_trace(model_ls,formatt='%.2f'):
    """
    input a list of SVM model, plot the roc_auc trace
    """
    C_ls = [model.C for model in model_ls]
    auc_ls = [model.auc for model in model_ls]
    plt.figure(figsize=(0.6*len(C_ls),6))
    plt.plot(auc_ls);
    plt.title('auc to C',fontsize=18);
    plt.ylabel('Area Under Curve',fontsize=12);
    plt.xlabel('hyper-parameter : C',fontsize=12);
    plt.xticks(np.arange(0,len(auc_ls)),[formatt%f for f in C_ls],fontsize=11);
    plt.yticks(fontsize=12);
    

vec_aa_dict = {0: 'A', 1: 'V', 2: 'I', 3: 'L', 4: 'M', 5: 'F', 6: 'Y', 7: 'W', 8: 'S', 9: 'T', 10: 'N', 11: 'Q', 12: 'R', 13: 'H', 14: 'K', 15: 'D', 16: 'E', 17: 'C', 18: 'U', 19: 'G', 20: 'P', 21: '-',22:'&'}

aa_vec_dict = {'A': 0, 'V': 1, 'I': 2, 'L': 3, 'M': 4, 'F': 5, 'Y': 6, 'W': 7, 'S': 8, 'T': 9, 'N': 10, 'Q': 11, 'R': 12, 'H': 13, 'K': 14, 'D': 15, 'E': 16, 'C': 17, 'U': 18, 'G': 19, 'P': 20, '-': 21, 'B': 22, 'J': 22, 'X': 22, 'Z': 22}

def flip_dict(dic):
    key=list(dic.keys())
    values=list(dic.values())
    if len(values) != len(np.unique(values)):
        return dict(zip(values,key))
    else:
        raise ValueError('flipped dict key has multiple pointer')

ratio_of = lambda y : np.sum(y==1)/len(y)        

class SVM_omega(object):
    global aa_vec_dict
    def __init__(self,coef,encoder_dict=aa_vec_dict,threshold='scale',keep2raw=None):  
        global aa_vec_dict
        global vec_aa_dict
        self.coef = coef[0]
        self.n_alphabet = len(np.unique(list(encoder_dict.values())))
        self.seq_len = len(self.coef)/self.n_alphabet
        self.threshold = np.quantile(np.abs(self.coef[np.where(self.coef != 0)[0]]),0.85) if threshold=='scale' else threshold
        self.aa2int = encoder_dict
        self.int2aa = vec_aa_dict if encoder_dict is aa_vec_dict else flip_dict(self.aa2int)
        self.important_sites = np.where(np.abs(self.coef)>self.threshold)[0]
        self.sites_weight = self.coef[self.important_sites]
        self.sites_in_seq = self.important_sites%self.seq_len
        self.sites_position = self.important_sites//self.seq_len       
        self.sites_aa = [self.int2aa[i] for i in self.sites_position]
        self.DF = self.dataframe()
        
        if len(self.important_sites) == 0:
            self.non_zero_hist()
            raise ValueError('Invalid threshold, please rechoose a threshold or try "scale"')
        
        if keep2raw is not None:
            self.raw_sites=np.array(keep2raw)[self.sites_in_seq.astype(int)]
            self.DF = self.dataframe(keep2raw=keep2raw)
                        
    def non_zero_hist(self,bins=20,**kwarg):
        plt.figure(figsize=(8,6))
        plt.hist(self.coef[np.where(self.coef != 0)[0]],bins=bins,**kwarg)
        
    def dataframe(self,columns=['sites_in_seq','sites_aa', 'sites_weight','sites_position'],keep2raw=None):
        if keep2raw is not None:
            columns=['raw_sites','sites_aa', 'sites_weight','sites_position']
        dataframe=pd.DataFrame(dict(zip(columns,[self.__getattribute__(x) for x in columns]))).sort_values(columns[0])
        dataframe.index = range(dataframe.shape[0])
        return dataframe
    
    def letter_plot(self,cmap=cm.cubehelix,**kwarg):
        figsize=(self.DF.shape[0]/4,3)
        plt.figure(figsize=figsize)
        plt.plot(range(self.DF.shape[0]),[1]*self.DF.shape[0],color='white')
        for i in range(self.DF.shape[0]):
            plt.text(i,1,self.DF.iloc[i,1],size=13+2*self.DF.iloc[i,2],color=cmap(self.DF.iloc[i,3]/22),**kwarg)
        