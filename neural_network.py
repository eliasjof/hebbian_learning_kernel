# -*- coding: utf-8 -*-
"""
Library to run different neural network 

Author: Elias J R Freitas
Date Created: 2021
Python Version: 3.8

"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import seaborn as sns 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.linalg import pinv
import sklearn
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class Processing:
    def __init__(self, metric="acc", Nstatistical=10, split=10):
        self.metric = metric
        self.Nstatistical = Nstatistical
        self.split  = split
    
    def run(self, clf, Xall, Yall, debug=1):
        
        ALL_test = []
        ALL_train = []

        for idxdata in range(len(Xall)):
            if debug:
                print("dataset ", idxdata)

            X = Xall[idxdata]
            Y = Yall[idxdata]
            
            results_train = []
            results_test = []
            clf_c = clf.copy()
            for clfi in clf_c: 
                mean_accuracy_test_clf = []
                mean_accuracy_train_clf=[]
                v_accuracy_test_clf = []
                v_accuracy_train_clf=[]

                if debug:
                    print("h -> ", clfi.hidden_size)

                
                for ist in range(self.Nstatistical):
                    
                    accuracy_test_kf = []
                    accuracy_train_kf=[]

                    kf =  ShuffleSplit(n_splits=self.split, test_size=0.25)
                    
                    for train_index, test_index in kf.split(X):
                        xtrain, xtest = X[train_index], X[test_index]
                        ytrain, ytest = Y[train_index], Y[test_index]  
                        # realiza o treinamento da rede
                        clfi.train(xtrain,ytrain)            
                        

                        
                        ##############
                        #resultado do teste
                        yhat_t = clfi.predict(xtest) 

                       
                        if self.metric =='acc':
                            metric = accuracy_score(ytest, yhat_t)
                        elif self.metric == 'auc':
                            fpr, tpr, _ = roc_curve(ytest, yhat_t, pos_label=1)
                            metric = auc(fpr, tpr)
                        elif self.metric=='mse':
                            metric = mean_squared_error(ytest, yhat_t,squared = True)
                        elif self.metric=='r2':
                            metric = r2_score(ytest, yhat_t)
                            
                        accuracy_test_kf.append(metric)
                        # print(metric)
                        ##############

                    if debug:
                        print("\t metric = ", np.mean(accuracy_test_kf))

                    
                    mean_accuracy_test_clf.append(np.mean(accuracy_test_kf))            
                    
                    v_accuracy_test_clf.append(np.var(accuracy_test_kf)**0.5)
                    
                results_test.append(mean_accuracy_test_clf)
                         
            
            ALL_test.append(results_test)
            # ALL_train.append(results_train)
            
        if debug:    
            print("fim")
        return ALL_train, ALL_test, clf_c

    def view_validation(self, data, data_names, h_size, save=False, name_fig=""):
        h_best = []
        metric_best = []
        ax_list = []
        for RT, name in zip(data, data_names):
            h_size2 = list(h_size)
            
            metric_mean = np.mean(RT, axis=1)
            mse_var = np.var(RT, axis=1)**0.5
            
            
            # print(metric_mean, mse_var)
            lower_bound = np.array(metric_mean) - np.array(mse_var)
            upper_bound = np.array(metric_mean) + np.array(mse_var)
            
            
            
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            ax.plot(h_size2, metric_mean,'-xb',label='mean')
            ax.fill_between(h_size2, lower_bound, upper_bound, facecolor='b', alpha=0.5,label='$\pm 1 \sigma$')
            
            # ax.set_xticks(h_size)
            ax.set_xlabel('neurônios')
            if(self.metric=='acc'):
                ax.set_ylabel('ACC (média $\pm$ desvio)')
                idx = np.argmax(metric_mean)
                h_best.append(h_size2[idx])
                metric_best.append(metric_mean[idx])
                label_best = 'best (h= '+str(h_best[-1]) + ", acc = "+ str(np.round(metric_best[-1],3)) + "$\pm$ " + str(np.round(mse_var[idx],3)) +")"
                ax.axis([0,h_size2[-1],0,1.1])
            elif self.metric=='auc':
                ax.set_ylabel('AUC (média $\pm$ desvio)')
                idx = np.argmax(metric_mean)
                h_best.append(h_size2[idx])
                metric_best.append(metric_mean[idx])
                label_best = 'best (h= '+str(h_best[-1]) + ", auc = "+ str(np.round(metric_best[-1],3)) + "$\pm$ " + str(np.round(mse_var[idx],3)) +")"
                ax.axis([0,h_size2[-1],0,1.1])
            elif self.metric=='mse':
                ax.set_ylabel('MSE (média $\pm$ desvio)')
                idx = np.argmin(metric_mean)
                h_best.append(h_size2[idx])
                metric_best.append(metric_mean[idx])
                label_best = 'best (h= '+str(h_best[-1]) + ", mse = "+ str(np.round(metric_best[-1],3)) + "$\pm$ " + str(np.round(mse_var[idx],3)) +")"
                ax.axis([0,h_size2[-1],0,0.1])
            elif self.metric=='r2':
                ax.set_ylabel('$R^2$ (média $\pm$ desvio)')
                idx = np.argmax(metric_mean)
                h_best.append(h_size2[idx])
                metric_best.append(metric_mean[idx])
                label_best = 'best (h= '+str(h_best[-1]) + ", $R^2$ = "+ str(np.round(metric_best[-1],3)) + "$\pm$ " + str(np.round(mse_var[idx],3)) +")"
                ax.axis([0,h_size2[-1],0,1.1])
            
            
            ax.plot(h_best[-1], metric_best[-1],'8r',label=label_best)
            ax.set_title(name)
            ax.grid(True)            
            
            ax.legend()
            ax_list.append(ax)
            
            fig.show()

            if(save):
                plt.savefig(name_fig+name+".png", bbox_inches = 'tight', pad_inches = 0)
        return metric_best, h_best, ax_list

class RBF(BaseEstimator):
    """ Class RBF """
    def __init__(self, hidden_size=3, sigma=None, lambda_reg=0.1, rbf='gaussian',type_f='unit', regularization=False, selection='kmeans'):
        self.hidden_size = hidden_size # número de neurônios na camada escondida 
        if type(sigma)== float: #raio fixo
            self.sigma = sigma*np.ones((hidden_size,1))
        else:
            self.sigma = sigma         
        self.rbf = rbf
        self.lambda_reg = lambda_reg
        self.selection = selection
        self.type_f = type_f #especifica a funçao de ativação
        self.regularization = regularization
        

        
        self.name = "RBF"
        
        if regularization:
            self.name += "with regularization " + "- lambda = " + str(lambda_reg)
        
        self.name += " (hidden_size = " + str(self.hidden_size) + ", function = "+str(type_f) + ", selection = " + str(selection) + ")"
        

        self.mu = [] #centros da função radial
        self.w = [] #pesos da camada de saída 

    
    def select_centers(self, X, max_iterations=15): 
        """ implementação naive do KMeans
        """

        if(self.selection=='kmeans'): 
            
            cluster = self.hidden_size if self.hidden_size < X.shape[0] else X.shape[0]
            kmeans = KMeans(n_clusters=cluster).fit(X)
            self.mu = kmeans.cluster_centers_     

            # print(self.mu.shape)     
            if self.sigma is None:
                N = X.shape[0]
                n = X.shape[1]
                p = self.hidden_size

            

                center = []
                for xt in X:
                    dist = []
                    for muj in self.mu:                    
                        d = np.linalg.norm(xt - muj)
                        dist.append(d)
                        
                    idx = np.argsort(dist)
                    center.append(idx[0])
            
                sigma=[]
                for i in range(p):
                    # obtem os indices das amostras que estão mais próximas do centro i
                    idxc = [idc for idc, ci in enumerate(center) if ci== i ]
                    Xc = X[idxc]
                    
                    
                    
                    # raio será dado pela covariância dos clusters
                    # print(Xc.shape,np.cov(Xc,rowvar=False).shape)
                    s = np.asarray(np.var(Xc))  if Xc.shape[0]==1 else np.mean(np.diag(np.cov(Xc,rowvar=False)))
                    # if s==0:
                    #     s=0.01
                    # else:
                    # print(1/len(X)*s)
                    sigma.append(1/len(X)*s +  0.01)
                    # sigma.append(s)   
            
            
                self.sigma = np.array(sigma) 
                self.sigma = self.sigma[:,None]
                
            
            
                
                
        else:
            # seleção randômica dos centros
            # Centro é a média entre dois pontos escolhidos aleatoriamente e
            # o vetor de raios aleatórios é obtido pela distância entre esses pontos.
            idx = np.random.permutation(len(X))
            pts1 = X[idx[:self.hidden_size]]
            idx2 = np.random.permutation(len(X))
            pts2 = X[idx2[:self.hidden_size]]   

            self.mu = (pts1 - pts2)/2

            sigma = []# np.array((self.hidden_size,1))
            
            for i in range(self.hidden_size):
                # print("i = ", i, 0.1*np.linalg.norm(pts1[i] + pts2[i]) + np.fabs(np.random.rand())/100)
                sigma.append( 1/len(X)*np.linalg.norm(pts1[i] + pts2[i]) )
                
                
            self.sigma=np.array(sigma)[:,None]**0.5
    
    def radial_function(self, x, mu, sigma):
        if self.rbf == 'gaussian':
            hj = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-sigma*np.linalg.norm(x-mu)**2)
            return hj

    def projection_matrix(self, X, debug=0):
        H = np.zeros((len(X), self.hidden_size))
        for i, xi in enumerate(X):
            for j, mu_j in enumerate(self.mu):                
                H[i, j] = self.radial_function(xi, mu_j, self.sigma[j][0])
        
        
        self.H = H
        return H


    def activation_function(self, x, w, type_f='tanh'):
        if type_f == 'unit':      
            return np.dot(x, w) # obtem o valor estimado da saída
        elif type_f == 'tanh':      
            return np.tanh(np.dot(x, w))
        elif type_f=='step':     
            y =  np.dot(x, w)    
            return np.sign(y) #np.where( y >= 0.5, 1.0, 0.0)  
        elif type_f=='tanh-class':     
            y =  np.tanh(np.dot(x, w)) 
            return np.sign(y) #np.where( y >= 0.0, 1.0, 0.0)  
        

    def fit(self, X, yd):
        
        self.select_centers(X) #seleciona o centro das funções radiais

        N = X.shape[0] # N amostras
        
        H = self.projection_matrix(X) 
        # H = np.hstack((H, np.ones((N,1))) )         

        # yd= yd[:, None] #garante que yi é um vetor
        if self.regularization: #regularização usando lambda
            m = 10^-16
            A = H.T.dot(H) + self.lambda_reg*np.eye(H.shape[1]) #+ np.eye(H.shape[1])*m
            
            self.w = np.linalg.pinv(A).dot(H.T).dot(yd)       
            
        else:
            # m = 10^-16
            # A = H.T.dot(H) + np.eye(H.shape[1])*m
            # A = H.T.dot(H)
            self.w = pinv(H).dot(yd)
            # self.w = np.linalg.inv(A).dot(H.T).dot(yd)  
        
        # self.w = np.linalg.inv(A).dot(H.T).dot(yd)  
            
        

    def predict(self, *args):   
        if len(args) == 1:
            X = args[0]
            H = self.projection_matrix(X)
            w = self.w

        elif len(args) == 2:
            X = args[0]
            w = args[1]
        
        H = self.projection_matrix(X)
        Y = self.activation_function(H, w, self.type_f) # função de mapeamento 
        return Y  




class ELM(BaseEstimator , ClassifierMixin):
    """ Class ELM """
    def __init__(self, hidden_size=10, limits=(-1.0, 1.0), type_map="random", tolerance=1e-3, learning_rate=0.001,max_iterations=1000, regularization=False, lambda_reg=0.1, type_f='tanh-class', n_outputs=1):
        """ 
        hidden_size # número de neurônios na camada escondida 
        type_map = random -> pesos obtidos randomicamente
                 hebb -> pesos obtidos por aprendizado Hebbiano
        """
        self.hidden_size = hidden_size # número de neurônios na camada escondida 
        self.limits = limits        
        self.type_map = type_map
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.type_f = type_f
        self.n_outputs = n_outputs

        
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def set_values(self):
        self.learned_ = False

        self.z_ = [] #pesos da camada escondida
        self.w_ = [] #pesos da camada de saída   

        self.name_ = "ELM"

        if self.type_map=="hebb":
            self.name_ += " with Hebbian Learning"
            self.name_ += "\n (hidden_size = " + str(self.hidden_size) +  ")"
        
        elif self.type_map=="hebb2":
            self.name_ += " with Generalized Hebbian Learning " + "\n (learning_rate = " + str(self.learning_rate)
            self.name_ += " hidden_size = " + str(self.hidden_size) +  ")"
        elif self.regularization:
            self.name_ += " with regularization"
        else:
            self.name_ += "\n (hidden_size = " + str(self.hidden_size) +  ")"


    def activation_function(self, x, w, type_f='tanh'):
        if type_f == 'unit':      
            return np.dot(x, w) # obtem o valor estimado da saída
        elif type_f == 'tanh':      
            return np.tanh(np.dot(x, w))
        elif type_f=='step':     
            y =  np.dot(x, w)    
            return np.sign(y) #np.where( y >= 0.5, 1.0, 0.0)  
        elif type_f=='tanh-class':     
            y =  np.tanh(np.dot(x, w)) 
            return np.sign(y) #np.where( y >= 0.0, 1.0, 0.0)  
             

    

    def learning_hebb(self, X, yd, w):
        N = X.shape[0]
        d = X.shape[1]
        
        
        prev_w = np.ones((1, self.hidden_size))
        # print("w = ", w[0])
        it = 0
        while np.linalg.norm(prev_w - w) > self.tolerance and it < self.max_iterations:   
            it +=1 
            # print(it)        
            # print(np.linalg.norm(prev_w - w))
            prev_w = w.copy()
            for n in range(N):
                yk = X[n].dot(w)
                    
                for i in range(d):
                    delta_w = self.learning_rate*yk*(X[n][i] - w[i].dot(yk))
                    w[i] += delta_w  
                    
                
        
        return w
    

    def fit(self, X, y):
        yd = y
        
        self.set_values()
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        
        N = X.shape[0] # N amostras   
        if len(X.shape) > 1:
                n = X.shape[1] # n entradas
        else:
            n = 1 

        
        X = np.hstack((X, np.ones((N,1))) ) 
                    
        
        
        # inicializa aleatoriamente os pesos da camada escondida        
        self.z_ = np.random.uniform(self.limits[0], self.limits[1],(n+1,self.hidden_size))   
        if self.type_map == "hebb2":
            self.z_ = self.learning_hebb(X, yd, self.z_)

        H = self.activation_function(X,self.z_,type_f='tanh') # função de mapeamento

        if self.regularization:            
            # m = 10^-12
            self.A = H.T.dot(H) + self.lambda_reg*np.eye(H.shape[1]) #+ np.eye(H.shape[1])*m
            self.w_ = np.linalg.inv(self.A).dot(H.T).dot(yd) 
        
        elif self.type_map == "hebb":            
            self.w_ = yd.T.dot(H)/np.linalg.norm(yd.T.dot(H))
            self.w_ = self.w_.T
        else: 
            Hplus = pinv(H )
            self.w_ = Hplus.dot(yd)
        
        self.learned_ = True

        # self.X_ = X
        # self.y_ = y
        return self
        

    def predict(self, X):
        """
        unit = Retorna a função diretamente x*w
        step = Retornar valores binaros 1 ou -1
        
        """
        check_is_fitted(self)
        X = check_array(X)
        X = np.hstack((X, np.ones((X.shape[0],1))) )  
        
        H = self.activation_function(X,self.z_,type_f='tanh') # função de mapeamento            
        return self.activation_function(H, self.w_, type_f=self.type_f)

    def _forward_pass_fast(self, X):       
        
        X = np.hstack((X, np.ones((X.shape[0],1))) )  
        
        H = self.activation_function(X,self.z_,type_f='tanh') # função de mapeamento            
        return self.activation_function(H, self.w_, type_f=self.type_f)

    
    def predict_proba(self,X):        
        check_is_fitted(self)
        X = np.hstack((X, np.ones((X.shape[0],1))) )  
        
        H = self.activation_function(X,self.z_,type_f='tanh') # função de mapeamento            
        y_pred =  self.activation_function(H, self.w_, type_f='tanh')
        

        if self.n_outputs == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred
        


class Perceptron(BaseEstimator):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.01, n_iter=200, tol=0.001, n_outputs=1, debug=False,type_='perceptron'):
        self.eta = eta
        self.n_iter = n_iter
        self.n_outputs = n_outputs
        self.tol = tol
        self.debug = debug
        self.type_ = type_
    def score(self, X, y, sample_weight=None):
        
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """
        
        X, y = check_X_y(X, y)
        y = y[:,None]
        
        self.classes_ = unique_labels(y)
        
        # print("class = ", self.classes_)

        
        self.error_ = []

        epoch_ = 0
        
        self.epoch_ = [0]
        if self.type_ == 'hebb':
                self.w_ = np.zeros((X.shape[1]+1,1))
                X = np.hstack((X, np.ones(( X.shape[0],1))) ) 
                self.w_ = y.T.dot(X)/np.linalg.norm(y.T.dot(X))
                self.w_ = self.w_.T
                yhat = self.predict(X)
        elif self.type_ ==  'pseudoinverse':
                self.w_ = np.zeros((X.shape[1]+1,1))
                X = np.hstack((X, np.ones(( X.shape[0],1))) ) 
                self.w_ = pinv(X).dot(y)
                # self.w_ = self.w_.T
                yhat = self.predict(X)

        else:
            self.w_ = np.zeros((X.shape[1]+1,1))
            X = np.hstack((X, np.ones(( X.shape[0],1))) ) 
            while True:
                yhat = self.predict(X)
                e = y - yhat
                
                self.w_ = self.w_ + self.eta*X.T.dot(e)

                epoch_+=1
                if self.debug:
                    print("e = ", np.linalg.norm(e)/X.shape[0])
                
                self.error_.append(np.linalg.norm(e)/X.shape[0])
                self.epoch_.append(epoch_)
                
                if np.linalg.norm(e) < self.tol or epoch_ >= self.n_iter:                    
                    break
            
        if self.debug:
            print('epoch_ = ', epoch_, 'e = ', np.linalg.norm(y - yhat)/X.shape[0])
            print('w = ', self.w_)
        # self.X_ = X
        # self.y_ = y
        return self

        

    def predict(self, X):
        """Return class label after unit step"""      
        if X.shape[1] < self.w_.shape[0]:
            X = np.hstack((X, np.ones(( X.shape[0],1))) )   

        # y_pred = np.sign(np.tanh(np.dot(X, self.w_)))
        y_pred = np.sign(np.dot(X, self.w_))
        # print("aqui = ", y_pred.shape,X.shape, self.w_.shape)
        return y_pred
        #return np.sign(np.dot(X, self.w_[1:]) + self.w_[0])
    
    def predict_proba(self,X):  
        X = np.hstack((X, np.ones(( X.shape[0],1))) ) 
        y_pred = np.sign(np.dot(X, self.w_) ) 
        # y_pred = np.sign(np.dot(X, self.w_[1:]) + self.w_[0])

        if self.n_outputs == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

class Adaline(BaseEstimator):
    def __init__(self, eta = 0.001, max_epoch = 100, tol=0.01, seed=100, limits=[-1, 1]):
        self.eta = eta
        self.max_epoch = max_epoch
        self.tol = tol
        self.seed = seed
        self.limits = limits
        self.error_ = []
        self.w = []

    def activation_function(self, x, w, type_f='unit'):
        if type_f == 'unit':      
            return np.dot(x, w) # obtem o valor estimado da saída
        if type_f=='step':
            return np.where(self.activation_function(x, w) >= 0.0, 1, -1)      



    def fit(self, X, yd, par=1):
        # np.random.seed(self.seed)        
        N = X.shape[0] # N amostras   
        if len(X.shape) > 1:
                n = X.shape[1] # n entradas
        else:
            n = 1 

        if par==1: # Adicionar coluna de 1's
            X = np.hstack((X, np.ones((N,1))) )        

        # inicializa aleatoriamente os valores de w
        if self.limits[0]== self.limits[1]:
            self.w = np.zeros((X.shape[1],1))
        else:
            self.w = np.random.uniform(self.limits[0], self.limits[1],(n+1,1))
        
        self.error_ = []
        self.epoch_ = [0]
        cost = 0
        
        while True:
            ei2 = 0 # erro quadrático relativo ao padrão i
            vselec = np.random.permutation(N) #vetor de seleção aleatório
            for i in range(0,N):            
                # escolhe um padrão
                index_Xi = vselec[i]            
                xi = X[index_Xi] #padrão de entrada escolhido            
                xi = xi[None,:] # força numpy a criar um vetor linha            
                
                if np.isnan(np.dot(self.w,self.w.T)[0][0]):
                    print("Erro ISNAN: epoca = ", self.epoch_[-1])                    
                    return
                else:
                    yhati = self.activation_function(xi, self.w)                   
                    

                    ei = yd[index_Xi] - yhati # calcula o erro do padrão i
                    
                    dw = self.eta*np.multiply(xi,ei) #Regra delta
                    
                    self.w = self.w + dw.T #atualiza o vetor de pesos
                    
                    ei2 += np.dot(ei,ei.T) #erro quadrático
                    
            self.error_.append(ei2[0][0]/N)
            self.epoch_.append(self.epoch_[-1]+1)        
        
            if self.error_[-1] < self.tol or self.epoch_[-1] >= self.max_epoch:
                print('epoca = ', self.epoch_[-1], 'e = ', self.error_[-1])
                break   

    def predict(self, X, type_f='unit'):
        """
        unit = Retorna a função diretamente x*w
        step = Retornar valores binaros 1 ou -1
        
        """
        
        X = np.hstack((X, np.ones((X.shape[0],1))) )  
        return self.activation_function(X, self.w)