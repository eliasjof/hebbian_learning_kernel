#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
################################################
"""
Hebbian learning Kernel-based (experimental tests)

Author: Elias J R Freitas
Date Updated: 2023
Python Version: >3.8


Usage:
1. Set the global variables to SAVE, to select the used classifiers and the datasets
2. Import this module into your Python script.
2. Initialize the optimization problem by defining the objective function, constraints, and other parameters.
3. Call the `l_shade` function to perform optimization. Pass the problem parameters as arguments.
4. Retrieve the optimized solution and other relevant information.

Example:
    $ ./python test_hebbian.py

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.linalg import pinv
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans

from load_datasets import *
from neural_network import *
from sklearn.model_selection import train_test_split
from kerneloptimizer.optimizer import KernelOptimizer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_predict, learning_curve,cross_val_score, cross_validate,GridSearchCV
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, RepeatedStratifiedKFold

from sklearn.utils.estimator_checks import check_estimator

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import csv
import time

# ## DATASETS
datasets =['australian','banknote','breastcancer','breastHess','bupa','climate','diabetes','fertility','german','glass','golub','haberman','heart','ilpd','ionosphere','parkinsons','sonar','spirals']
# datasets = ['australian']
path = './results/paper/'

SAVE = 1
run_MLP = 1
run_ELM = 1
run_SVM = 1

ds = DatasetsLoad()
with open(path + '_datasets_info.txt','+w') as f:
    f.writelines('base \t parameters \t size base \t class 1 (%) class 2 (%)\n')
    
for ii,dataset in enumerate(datasets):
    X, Y, _ = ds.load(dataset)
    count = np.count_nonzero(Y != 1.)
    with open(path + '_datasets_info.txt','a') as f: # The with keyword automatically closes the file when you are done
        f.writelines('{} \t {} \t {} \t {} \t {}\n'.format(dataset, X.shape[1],Y.shape[0],np.round(count/Y.shape[0]*100,2), 100 - np.round(count/Y.shape[0]*100,2)))


""" KMLPH e KMLPA """
for ii,dataset in enumerate(datasets):
    print(ii)    
    X, Y, _ = ds.load(dataset) 
    Nstatistical = 30

    acc_ADALINE = []
    auc_ADALINE = []

    acc_HEBB = []
    auc_HEBB = []

    acc_KMLPH = []
    auc_KMLPH = []

    acc_KMLPA = []
    auc_KMLPA = []

    acc_KGAUSSA = []
    auc_KGAUSSA = []

    acc_KGAUSSH = []
    auc_KGAUSSH = []

    acc_SVM = []
    auc_SVM = []

    acc_MLP = []
    auc_MLP = []

    acc_ELM = []
    auc_ELM = []

    time_HEBB = []
    time_ADALINE = []
    time_KGAUSSH = []
    time_KGAUSSA = []
    time_KMLPA = []
    time_KMLPH = []
    time_SVM = []
    time_MLP = []
    time_ELM = []

    for i in range(Nstatistical):
    # kf = StratifiedKFold(n_splits=10)    
    # kf = KFold(n_splits=10)    
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, stratify=Y)
    # for train_index, test_index in kf.split(X,Y):
    # for train_index, test_index in kf.split(X):
        
        # xtrain, xtest = X[train_index], X[test_index]
        # ytrain, ytest = Y[train_index], Y[test_index]
        
        ''' ADALINE '''
        init = time.time()
        clf_ADALINE = Perceptron(type_='delta')        
        clf_ADALINE.fit(xtrain, ytrain) 
        yhat = clf_ADALINE.predict(xtest)

        t_elapsed = time.time() - init
        time_ADALINE.append(t_elapsed)

        acc = accuracy_score(yhat , ytest)
        fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
        auc_value = auc(fpr, tpr)
        acc_ADALINE.append(acc)
        auc_ADALINE.append(auc_value)

        ''' HEBB '''
        init = time.time()
        clf_HEBB = Perceptron(type_='hebb')        
        clf_HEBB.fit(xtrain, ytrain) 
        yhat = clf_HEBB.predict(xtest)
        t_elapsed = time.time() - init
        time_HEBB.append(t_elapsed)

        acc = accuracy_score(yhat , ytest)
        fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
        auc_value = auc(fpr, tpr)

        acc_HEBB.append(acc)
        auc_HEBB.append(auc_value)

        ''' MAPEAMENTO KERNEL '''
        init = time.time()

        Phi_g = KernelOptimizer(
        kernel='gaussian',
        input_dim=X.shape[1],
        hidden_dim=40,
        output_dim=100)

        Phi_g.fit(xtrain,ytrain,n_epochs=1000, lr=0.01)
        lspace_g = Phi_g.get_likelihood_space(xtrain).to_numpy()   
        end_phig = time.time() - init #time for obtained likelihood space
        
        init = time.time()

        Phi = KernelOptimizer(
        kernel='mlp',
        input_dim=X.shape[1],
        hidden_dim=40,
        output_dim=100)

        Phi.fit(xtrain,ytrain,n_epochs=1000, lr=0.01)
        lspace = Phi.get_likelihood_space(xtrain).to_numpy()  

        end_phi = time.time() - init #time for obtained likelihood space
             

        ''' KMLP - HEBBIAN '''
        init = time.time()
        mu = np.mean(lspace, axis=0)
        sigma = np.std(lspace, axis=0)
        lspace_H = (lspace - mu)/sigma        
        clf_KMLPH = Perceptron(type_='hebb')        
        clf_KMLPH.fit(lspace_H, ytrain)
        lspace_H = Phi.get_likelihood_space(xtest).to_numpy()      
        lspace_H = (lspace_H - mu)/sigma  
        yhat = clf_KMLPH.predict(lspace_H)

        t_elapsed = time.time() - init + end_phi
        time_KMLPH.append(t_elapsed)        
              
        acc = accuracy_score(yhat , ytest)
        fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
        auc_value = auc(fpr, tpr)

        acc_KMLPH.append(acc)
        auc_KMLPH.append(auc_value)
        print(f'KMLP-H acc = {acc}, auc = {auc_value}')

        ''' KMLP - Adaline '''    
        init = time.time()    
        clf_KMLPA = Perceptron(type_='delta')        
        clf_KMLPA.fit(lspace, ytrain)
        lspace_A = Phi.get_likelihood_space(xtest).to_numpy() 
        yhat = clf_KMLPA.predict(lspace_A)

        t_elapsed = time.time() - init + end_phi
        time_KMLPA.append(t_elapsed)       
              
        acc = accuracy_score(yhat , ytest)
        fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
        auc_value = auc(fpr, tpr)

        acc_KMLPA.append(acc)
        auc_KMLPA.append(auc_value)


        ''' KGAUSS - HEBBIAN '''
        init = time.time()
        mu = np.mean(lspace_g, axis=0)
        sigma = np.std(lspace_g, axis=0)
        lspace_H_g = (lspace_g - mu)/sigma        
        clf_KGAUSSH = Perceptron(type_='hebb')        
        clf_KGAUSSH.fit(lspace_H_g, ytrain)
        lspace_H_g = Phi_g.get_likelihood_space(xtest).to_numpy() 
        lspace_H_g = (lspace_H_g - mu)/sigma  
        yhat = clf_KGAUSSH.predict(lspace_H_g)

        t_elapsed = time.time() - init + end_phig
        time_KGAUSSH.append(t_elapsed)       
              
        acc = accuracy_score(yhat , ytest)
        fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
        auc_value = auc(fpr, tpr)

        acc_KGAUSSH.append(acc)
        auc_KGAUSSH.append(auc_value)
        print(f'KRBF-H acc = {acc}, auc = {auc_value}')

        ''' KGAUSS - Adaline '''     
        init = time.time()   
        clf_KGAUSSA = Perceptron(type_='delta')        
        clf_KGAUSSA.fit(lspace_g, ytrain)
        lspace_A = Phi_g.get_likelihood_space(xtest).to_numpy() 
        yhat = clf_KGAUSSA.predict(lspace_A)

        t_elapsed = time.time() - init + end_phig
        time_KGAUSSA.append(t_elapsed)       
              
        acc = accuracy_score(ytest, yhat)
        fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
        auc_value = auc(fpr, tpr)

        acc_KGAUSSA.append(acc)
        auc_KGAUSSA.append(auc_value)

        if run_SVM:
            ''' SVM '''
            print('SVM')
            init = time.time()
            Gamma = np.logspace(-2, 1, base=10, num=10*2)
            C = list(np.logspace(-1, 2, base=10, num=10*2))
                
            Kernel = ['rbf']
            kfold = StratifiedKFold(n_splits = 10)
            
            svm_ = svm.SVC()
            parameters = {'kernel':Kernel, 'gamma':Gamma, 'C':C}
            clf = GridSearchCV(svm_, parameters, cv=StratifiedKFold(n_splits = 10),scoring='accuracy')
            clf.fit(xtrain, ytrain)
            Kernel_best = clf.best_params_['kernel']
            C_best_SVM = clf.best_params_['C']
            Gamma_best = clf.best_params_['gamma']
            # print(Gamma_best, C_best_SVM, Kernel_best)

            ''' Train and Test '''            
            clf2 = svm.SVC(C=C_best_SVM, kernel=Kernel_best,gamma=Gamma_best)
            clf2.fit(xtrain, ytrain)
            yhat = clf2.predict(xtest)

            t_elapsed = time.time() - init
            time_SVM.append(t_elapsed)       
                
            acc = accuracy_score(yhat , ytest)
            fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
            auc_value = auc(fpr, tpr)

            acc_SVM.append(acc)
            auc_SVM.append(auc_value)

        #########################################
        if run_MLP:
            ''' MLP '''
            print("MLP")
            init = time.time()
            kfold = StratifiedKFold(n_splits = 10)
            # H = range(10,1000,10*2)#list(np.logspace(1, 3, base=10, num=10*3)) #[500]#range(10,1000,10*3) #list(np.logspace(1, 3, base=10, num=10*3))
            # C = ['relu', 'logistic', 'tanh']
            # C =  list(np.logspace(-1, 2, base=10, num=10*2))
            H = range(5,200,10)
            C =  list(np.logspace(-4, 1, base=2, num=10))
            C.insert(0,0)
            mlp = MLPClassifier(learning_rate_init=1e-2, activation = 'tanh', max_iter=30000, solver='adam')
            parameters = {'alpha':C, 'hidden_layer_sizes':H}
            clf = GridSearchCV(mlp, parameters, cv=StratifiedKFold(n_splits = 10),scoring='accuracy')
            clf.fit(xtrain, ytrain)
            C_best_MLP = clf.best_params_['alpha']
            h_best_MLP = clf.best_params_['hidden_layer_sizes']

            C_best_MLP = clf.best_params_['alpha']
            h_best_MLP = clf.best_params_['hidden_layer_sizes']

            clf2 = MLPClassifier(alpha=C_best_MLP, activation = 'tanh', hidden_layer_sizes=(h_best_MLP,),learning_rate_init=1e-2, max_iter=30000, solver='adam')
            clf2.fit(xtrain, ytrain)
            yhat = clf2.predict(xtest)

            t_elapsed = time.time() - init
            time_MLP.append(t_elapsed)       
                
            acc = accuracy_score(yhat , ytest)
            fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
            auc_value = auc(fpr, tpr)

            acc_MLP.append(acc)
            auc_MLP.append(auc_value)
            print(f'MLP acc = {acc}, auc = {auc_value}')

        #########################################
        if run_ELM:
            ''' ELM '''
            print("ELM")
            init = time.time()
            kfold = StratifiedKFold(n_splits = 10)
            H  = range(5,200,10)#= np.logspace(1, 3, base=10, num=10*3) #[500] #range(10,1000,10*3) #np.logspace(1, 3, base=10, num=10*3)
            C =  np.logspace(-4, 1, base=2, num=10)
            # print(H)
            # C.insert(0,0)
            results_mean = []
            results_var = []
            kfold = KFold(n_splits = 10)
            for h in H:
                for c in C:                    
                    clf = ELM(regularization=True, hidden_size=int(h), lambda_reg=c)
                    score = cross_val_score(clf, xtrain, ytrain, scoring='accuracy', cv=kfold)
                    results_mean.append(np.mean(score))
                    results_var.append(np.var(score))
            
            idmax = np.argmax(results_mean)
            
            xx, yy = np.meshgrid(H, C)
            xx = np.reshape(xx.T,(-1,1))
            yy = np.reshape(yy.T,(-1,1))
            h_best_ELM = xx[idmax][0]
            C_best_ELM = yy[idmax][0]

            
            clf2 = ELM(regularization=True, hidden_size=h_best_ELM, lambda_reg=C_best_ELM)
            clf2.fit(xtrain, ytrain)
            yhat = clf2.predict(xtest)        

            t_elapsed = time.time() - init
            time_ELM.append(t_elapsed)       
                
            acc = accuracy_score(yhat , ytest)
            fpr, tpr, _ = roc_curve(ytest , yhat, pos_label=1)
            auc_value = auc(fpr, tpr)

            acc_ELM.append(acc)
            auc_ELM.append(auc_value)

            print(f'ELM acc = {acc}, auc = {auc_value}')

        if SAVE:
            if run_SVM:
                df = pd.read_excel(path+'_best_params.xlsx',engine='openpyxl')
                name_datasets = list(df['dataset'])
                df.loc[name_datasets.index(dataset),'SVM'] = 'gamma = ' + str(Gamma_best) + ' reg = ' + str(C_best_SVM) + ' kernel = ' + str(Kernel_best)
                df.to_excel(path+"_best_params.xlsx",index=False)  
            
            if run_MLP:
                df = pd.read_excel(path+'_best_params.xlsx',engine='openpyxl')
                name_datasets = list(df['dataset'])
                df.loc[name_datasets.index(dataset),'MLP'] = 'hidden_size = ' + str(h_best_MLP) + ' activation = ' + str(C_best_MLP)
                df.to_excel(path+"_best_params.xlsx",index=False)  

            if run_ELM:
                df = pd.read_excel(path+'_best_params.xlsx',engine='openpyxl')
                name_datasets = list(df['dataset'])
                df.loc[name_datasets.index(dataset),'ELM'] = 'hidden_size = ' + str(h_best_ELM) + ' lambda = ' + str(C_best_ELM)
                df.to_excel(path+"_best_params.xlsx",index=False)  
            

    # print("acc_ADALINE = ", np.mean(acc_ADALINE)*100, "+- ", (np.var(acc_ADALINE)**0.5)*100)
    # print("acc_HEBB= ", np.mean(acc_HEBB)*100, "+- ", (np.var(acc_HEBB)**0.5)*100)
    # print("acc_KMLPH= ", np.mean(acc_KMLPH)*100, "+- ", (np.var(acc_KMLPH)**0.5)*100)
    # print("acc_KMLPA = ",np.mean(acc_KMLPA)*100, "+- ", (np.var(acc_KMLPA)**0.5)*100)
    # print("acc_KGAUSSA= ", np.mean(acc_KGAUSSA)*100, "+- ", (np.var(acc_KGAUSSA)**0.5)*100)
    # print("acc_KGAUSSH = ",np.mean(acc_KGAUSSH)*100, "+- ", (np.var(acc_KGAUSSH)**0.5)*100)
    # print("acc_SVM = ",np.mean(acc_SVM)*100, "+- ", (np.var(acc_SVM)**0.5)*100)
    # print("acc_MLP = ",np.mean(acc_MLP)*100, "+- ", (np.var(acc_MLP)**0.5)*100)
    # print("acc_ELM = ",np.mean(acc_ELM)*100, "+- ", (np.var(acc_ELM)**0.5)*100)
    
    
    print("auc_ADALINE = ", np.mean(auc_ADALINE)*100, "+- ", (np.var(auc_ADALINE)**0.5)*100)
    print("auc_HEBB= ", np.mean(auc_HEBB)*100, "+- ", (np.var(auc_HEBB)**0.5)*100)
    print("auc_KMLPH= ", np.mean(auc_KMLPH)*100, "+- ", (np.var(auc_KMLPH)**0.5)*100)
    print("auc_KMLPA = ",np.mean(auc_KMLPA)*100, "+- ", (np.var(auc_KMLPA)**0.5)*100)
    print("auc_KGAUSSA= ", np.mean(auc_KGAUSSA)*100, "+- ", (np.var(auc_KGAUSSA)**0.5)*100)
    print("auc_KGAUSSH = ",np.mean(auc_KGAUSSH)*100, "+- ", (np.var(auc_KGAUSSH)**0.5)*100)
    
    # print(np.mean(time_HEBB), np.mean(time_ADALINE), np.mean(time_KGAUSSA), np.mean(time_KGAUSSH), np.mean(time_KMLPA), np.mean(time_KMLPH),np.mean(time_SVM), np.mean(time_MLP), np.mean(time_ELM))

    
    ''' SAVING DATA'''
    if SAVE:
        df = pd.read_excel(path + '_complete_result.xlsx',engine='openpyxl')
        name_datasets = list(df['dataset'])

        df.loc[name_datasets.index(dataset),'AUC_ADALINE'] = np.round(100*np.mean(auc_ADALINE),2)
        df.loc[name_datasets.index(dataset),'std_AUC_ADALINE'] = np.round(100*np.std(auc_ADALINE),2)
        df.loc[name_datasets.index(dataset),'ACC_ADALINE'] = np.round(100*np.mean(acc_ADALINE),2)
        df.loc[name_datasets.index(dataset),'std_ACC_ADALINE'] = np.round(100*np.std(acc_ADALINE),2)        

        df.loc[name_datasets.index(dataset),'AUC_HEBB'] = np.round(100*np.mean(auc_HEBB),2)
        df.loc[name_datasets.index(dataset),'std_AUC_HEBB'] = np.round(100*np.std(auc_HEBB),2)
        df.loc[name_datasets.index(dataset),'ACC_HEBB'] = np.round(100*np.mean(acc_HEBB),2)
        df.loc[name_datasets.index(dataset),'std_ACC_HEBB'] = np.round(100*np.std(acc_HEBB),2)

        df.loc[name_datasets.index(dataset),'AUC_KGAUSSA'] = np.round(100*np.mean(auc_KGAUSSA),2)
        df.loc[name_datasets.index(dataset),'std_AUC_KGAUSSA'] = np.round(100*np.std(auc_KGAUSSA),2)
        df.loc[name_datasets.index(dataset),'ACC_KGAUSSA'] = np.round(100*np.mean(acc_KGAUSSA),2)
        df.loc[name_datasets.index(dataset),'std_ACC_KGAUSSA'] = np.round(100*np.std(acc_KGAUSSA),2)  

        df.loc[name_datasets.index(dataset),'AUC_KGAUSSH'] = np.round(100*np.mean(auc_KGAUSSH),2)
        df.loc[name_datasets.index(dataset),'std_AUC_KGAUSSH'] = np.round(100*np.std(auc_KGAUSSH),2)
        df.loc[name_datasets.index(dataset),'ACC_KGAUSSH'] = np.round(100*np.mean(acc_KGAUSSH),2)
        df.loc[name_datasets.index(dataset),'std_ACC_KGAUSSH'] = np.round(100*np.std(acc_KGAUSSH),2)    

        df.loc[name_datasets.index(dataset),'AUC_KMLPA'] = np.round(100*np.mean(auc_KMLPA),2)
        df.loc[name_datasets.index(dataset),'std_AUC_KMLPA'] = np.round(100*np.std(auc_KMLPA),2)
        df.loc[name_datasets.index(dataset),'ACC_KMLPA'] = np.round(100*np.mean(acc_KMLPA),2)
        df.loc[name_datasets.index(dataset),'std_ACC_KMLPA'] = np.round(100*np.std(acc_KMLPA),2)    

        df.loc[name_datasets.index(dataset),'AUC_KMLPH'] = np.round(100*np.mean(auc_KMLPH),2)
        df.loc[name_datasets.index(dataset),'std_AUC_KMLPH'] = np.round(100*np.std(auc_KMLPH),2)
        df.loc[name_datasets.index(dataset),'ACC_KMLPH'] = np.round(100*np.mean(acc_KMLPH),2)
        df.loc[name_datasets.index(dataset),'std_ACC_KMLPH'] = np.round(100*np.std(acc_KMLPH),2)

        if run_SVM:
            df.loc[name_datasets.index(dataset),'AUC_SVM'] = np.round(100*np.mean(auc_SVM),2)
            df.loc[name_datasets.index(dataset),'std_AUC_SVM'] = np.round(100*np.std(auc_SVM),2)
            df.loc[name_datasets.index(dataset),'ACC_SVM'] = np.round(100*np.mean(acc_SVM),2)
            df.loc[name_datasets.index(dataset),'std_ACC_SVM'] = np.round(100*np.std(acc_SVM),2)
            
        if run_MLP:
            df.loc[name_datasets.index(dataset),'AUC_MLP'] = np.round(100*np.mean(auc_MLP),2)
            df.loc[name_datasets.index(dataset),'std_AUC_MLP'] = np.round(100*np.std(auc_MLP),2)
            df.loc[name_datasets.index(dataset),'ACC_MLP'] = np.round(100*np.mean(acc_MLP),2)
            df.loc[name_datasets.index(dataset),'std_ACC_MLP'] = np.round(100*np.std(acc_MLP),2)
        if run_ELM:
            df.loc[name_datasets.index(dataset),'AUC_ELM'] = np.round(100*np.mean(auc_ELM),2)
            df.loc[name_datasets.index(dataset),'std_AUC_ELM'] = np.round(100*np.std(auc_ELM),2)
            df.loc[name_datasets.index(dataset),'ACC_ELM'] = np.round(100*np.mean(acc_ELM),2)
            df.loc[name_datasets.index(dataset),'std_ACC_ELM'] = np.round(100*np.std(acc_ELM),2)
        
        df.loc[name_datasets.index(dataset),'time_HEBB'] = np.round(100*np.mean(time_HEBB),2)
        df.loc[name_datasets.index(dataset),'std_time_HEBB'] = np.round(100*np.std(time_HEBB),2)
        df.loc[name_datasets.index(dataset),'time_ADALINE'] = np.round(100*np.mean(time_ADALINE),2)
        df.loc[name_datasets.index(dataset),'std_time_ADALINE'] = np.round(100*np.std(time_ADALINE),2)
        df.loc[name_datasets.index(dataset),'time_KGAUSSH'] = np.round(100*np.mean(time_KGAUSSH),2)
        df.loc[name_datasets.index(dataset),'std_time_KGAUSSH'] = np.round(100*np.std(time_KGAUSSH),2)
        df.loc[name_datasets.index(dataset),'time_KGAUSSA'] = np.round(100*np.mean(time_KGAUSSA),2)
        df.loc[name_datasets.index(dataset),'std_time_KGAUSSA'] = np.round(100*np.std(time_KGAUSSA),2)
        df.loc[name_datasets.index(dataset),'time_KMLPA'] = np.round(100*np.mean(time_KMLPA),2)
        df.loc[name_datasets.index(dataset),'std_time_KMLPA'] = np.round(100*np.std(time_KMLPA),2)
        df.loc[name_datasets.index(dataset),'time_KMLPH'] = np.round(100*np.mean(time_KMLPH),2)
        df.loc[name_datasets.index(dataset),'std_time_KMLPH'] = np.round(100*np.std(time_KMLPH),2)
        
        if run_SVM:
            df.loc[name_datasets.index(dataset),'time_SVM'] = np.round(100*np.mean(time_SVM),2)
            df.loc[name_datasets.index(dataset),'std_time_SVM'] = np.round(100*np.std(time_SVM),2)
        
        # df.loc[name_datasets.index(dataset),'time_MLP'] = np.round(100*np.mean(time_MLP),2)
        # df.loc[name_datasets.index(dataset),'std_time_MLP'] = np.round(100*np.std(time_MLP),2) 
        # df.loc[name_datasets.index(dataset),'time_ELM'] = np.round(100*np.mean(time_ELM),2)
        # df.loc[name_datasets.index(dataset),'std_time_ELM'] = np.round(100*np.std(time_ELM),2) 
        
        df.to_excel(path + '_complete_result.xlsx',index=False)

        if run_SVM == 0:
            acc_SVM= list(np.zeros((1,len(auc_ADALINE)))[0])
            auc_SVM= list(np.zeros((1,len(auc_ADALINE)))[0])
            time_SVM = list(np.zeros((1,len(time_ADALINE)))[0])

        if run_MLP == 0:
            acc_MLP= list(np.zeros((1,len(auc_ADALINE)))[0])
            auc_MLP= list(np.zeros((1,len(auc_ADALINE)))[0])
            time_MLP = list(np.zeros((1,len(time_ADALINE)))[0])

        if run_ELM == 0:
            acc_ELM = list(np.zeros((1,len(auc_ADALINE)))[0])
            auc_ELM = list(np.zeros((1,len(auc_ADALINE)))[0])
            time_ELM = list(np.zeros((1,len(time_ADALINE)))[0])

        fields = ["ADALINE","HEBB","SVM","ELM","MLP","KGAUSSA","KGAUSSH","KMLPA","KMLPH"]
        rows_acc = []
        rows_auc = []
        for i in range(len(acc_KGAUSSA)):
            row_acc = [acc_ADALINE[i],acc_HEBB[i],acc_SVM[i],acc_ELM[i],acc_MLP[i],acc_KGAUSSA[i],acc_KGAUSSH[i],acc_KMLPA[i],acc_KMLPH[i]]
            row_auc = [auc_ADALINE[i],auc_HEBB[i],auc_SVM[i],auc_ELM[i],auc_MLP[i],auc_KGAUSSA[i],auc_KGAUSSH[i],auc_KMLPA[i],auc_KMLPH[i] ]

            rows_acc.append(row_acc)
            rows_auc.append(row_auc)

        with open(path+dataset+".csv", '+w') as f:
            write = csv.writer(f)            
            write.writerow(fields)
            write.writerows(rows_acc)

        with open(path+dataset+"_auc.csv", '+w') as f:
            write = csv.writer(f)            
            write.writerow(fields)
            write.writerows(rows_auc)

        rows_time = []
        for i in range(len(time_KMLPH)):
            rows_timei = [time_ADALINE[i],time_HEBB[i],time_SVM[i],time_ELM[i],time_MLP[i],time_KGAUSSA[i],time_KGAUSSH[i],time_KMLPA[i],time_KMLPH[i]]
            rows_time.append(rows_timei)
            
        with open(path+dataset+"_time.csv", '+w') as f:
            write = csv.writer(f) 
            write.writerow(fields)
            write.writerows(rows_time)
