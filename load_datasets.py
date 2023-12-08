# -*- coding: utf-8 -*-
"""
Library to load many datasets

Author: Elias J R Freitas
Date Created: 2021
Python Version: 3.8

Description:
load many datasets

Usage:
1. Import the library
2. Create the object
3. Load the desired dataset from ./dataset/ 

Example:
```
from load_datasets import *
ds = DatasetsLoad(name_path="./dataset/")
ds.load('name_dataset')
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer, load_diabetes
import sklearn
from sklearn.preprocessing import LabelEncoder

from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import scipy.io as sio


class DatasetsLoad:
    
    def __init__(self, name_path="./dataset/"):
        self.name_path = name_path
        self.datasets = {
            'banknote': self.banknote(),
            'breastcancer': self.breastcancer(),
            'climate': self.climate(),            
            'fertility': self.fertility(),
            'glass': self.glass(),
            'heart': self.heart(),
            'ionosphere': self.ionosphere(),
            'iris': self.iris(),
            'parkinsons': self.parkinsons(),
            'spirals': self.spirals(),
            'sonar': self.sonar(),
            'xor': self.xor(),            
            'automobile': self.automobile(),            
            'handwritten': self.handwritten(),            
            'spirals_n': self.spirals_n(),
            'spirals_t': self.spirals_t(),
            'wine': self.wine(),            
            'normal': self.normal(),
            'sinc': self.sinc(), 
            'w3a': self.w3a(), 
            'w3at': self.w3at(), 
            'ijcnn': self.ijcnn(),   
            'phishing': self.phishing(), 
            'ilpd': self.ilpd(),
            'australian': self.australian(),
            'breastHess': self.breastHess(),
            'bupa': self.bupa(),
            'german': self.german(),
            'golub': self.golub(),
            'haberman': self.haberman(),
            'diabetes': self.diabetes(),
            'forest': self.forest(),
            'pima': self.pima(),
        }

    def pima(self):
        data=pd.read_csv(self.name_path +'pima-indians-diabetes.csv')
        Y=data.iloc[:,8]
        Y = np.where(Y ==1,1.0, -1.0)  
        X = data.iloc[:,0:8]
        stdScaler_data = StandardScaler()
        X = stdScaler_data.fit_transform(X)
             
        return X, Y, data

    def forest(self):
        data=pd.read_csv(self.name_path +'covtype.csv')
        Y=data['Cover_Type']
        X = data.iloc[:,0:53]
        stdScaler_data = StandardScaler()
        X = stdScaler_data.fit_transform(X)
        Xo, X, Yo, Y = train_test_split(X, Y, test_size=0.01, stratify=Y)
        return X, Y, data


    def diabetes(self):
        dataset = 'diabetes'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat

    def haberman(self):
        dataset = 'haberman'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat
    
    
    def golub(self):
        dataset = 'golub'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat
    def german(self):
        dataset = 'german'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat
    
    def bupa(self):
        dataset = 'bupa'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat

    def breastHess(self):
        dataset = 'breastHess'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat

    def australian(self):
        dataset = 'australian'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat

    def ilpd(self):
        dataset = 'ILPD'
        fold_n = 0
        filename = "{}exportBase_{}_folds_10_exec_{}.mat".format(self.name_path,dataset, fold_n+1)
        data_mat = sio.loadmat(filename)
        train =  data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        X = np.vstack((test, train))
        Y = np.hstack((classTest, classTrain))

        return X, Y, data_mat
    
    def url(self):
        data = load_svmlight_file(self.name_path + "url_combined.bz2")
        X, Y = data[0], data[1]
        X = np.array(X.todense().tolist())

        return X, Y, data
    
    
    def phishing(self):
        data = load_svmlight_file(self.name_path + "phishing")
        X, Y = data[0], data[1]
        X = np.array(X.todense().tolist())

        return X, Y, data


    def w3a(self):
        data = load_svmlight_file(self.name_path + "w3a")
        X, Y = data[0], data[1]
        X = np.array(X.todense().tolist())

        return X, Y, data

    def w3at(self):
        data = load_svmlight_file(self.name_path + "w3a.t")
        X, Y = data[0], data[1]
        X = np.array(X.todense().tolist())

        return X, Y, data
    
    def ijcnn(self):
        data = load_svmlight_file(self.name_path + "ijcnn1.bz2")
        X, Y = data[0], data[1]
        X = np.array(X.todense().tolist())

        return X, Y, data

    def heart(self):
        data2 = pd.read_csv(self.name_path + 'heart-statlog.csv')  
        y = data2['class']
        Y = np.where(y =='present',1.0, -1.0)
        data2 = data2.drop(columns=['class'])
        X = data2.values
        stdScaler_data = StandardScaler()
        X = stdScaler_data.fit_transform(X)
        return X, Y, data2

    def parkinsons(self):
        """ https://www.kaggle.com/itsmesunil/parkinsons-disease
        """
        df = pd.read_csv(self.name_path + 'parkinsons.txt')
        Y = df['status'].values
        Y = np.where(Y==0,-1,1)
        df = df.drop(columns=['status', 'name'])
        X = df.to_numpy()
        ss = StandardScaler()
        X = ss.fit_transform(X)
        return X, Y, df
    
    def glass(self):
        """
        https://www.kaggle.com/mohamedzayton/glass-classification-rf
        """
        df = pd.read_csv(self.name_path + 'glass.csv')
        X = df.drop(columns= ['Type']).values
        Y = df.iloc[:, -1].values
        Y = np.where(Y==1,1,-1)
        ss = StandardScaler()
        X = ss.fit_transform(X)
        return X, Y, df
    
    def fertility(self):
        """ referência: https://www.kaggle.com/pranavkasela/automl-with-smac-and-sklearn
        """

        df = pd.read_csv(self.name_path + 'fertility.csv', skiprows=[0] , names=['Season','Age','Childish diseases','Accident or trauma','Surgical intervention','High fevers',               'Frequency of alcohol consumption','Smoking habit' , 'Hours spent sitting per day','Class'])
        df = pd.get_dummies(df, columns=['Season','Frequency of alcohol consumption'])

        encode_col =['Childish diseases','Accident or trauma',
                    'Surgical intervention','High fevers',
                    'Smoking habit', 'Hours spent sitting per day', 'Class']

        # converte labels para número
        for col in encode_col:
            le = LabelEncoder().fit(df[col])
            df[col] = le.transform(df[col])
            
        df.Class = 1 - df.Class
        Y = df.Class.to_numpy()
        Y = np.where(Y==0,-1,1)
        df = df.drop(columns=['Class'])
        X = df.to_numpy()

        ss = StandardScaler()
        X = ss.fit_transform(X)

        return X, Y, df

    def climate(self):
        df = pd.read_excel(self.name_path + 'climate.xlsx',engine='openpyxl')
        X = df.iloc[:,:-1].to_numpy()
        Y = df.iloc[:,-1].to_numpy()
        Y = np.where(Y==0,-1,1)
        ss = StandardScaler()
        X = ss.fit_transform(X)
        return X, Y, df

    def banknote(self):
        """ ver detalhes de referência: https://www.kaggle.com/prashansdixit/bank-note-authentication-svc-100-acc
        """

        df = pd.read_csv(self.name_path + 'BankNoteAuthentication.csv')
        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]
        Y = np.where(Y==0,-1,1)
        ss = StandardScaler()
        X = ss.fit_transform(X)
        
        return X, Y, df

    def xor(self):
        data_xor = pd.read_csv(self.name_path + 'xor2.csv')
        Y_xor = data_xor['classes']
        Y_xor = np.where( Y_xor == 1.0, 1.0, -1.0) 
        data_xor = data_xor.drop(columns=['classes'])
        X_xor = data_xor.values
        data_xor = np.hstack((X_xor,Y_xor[:,None]))
        return X_xor, Y_xor, data_xor
    
    def wine(self):
        """## Wine Quality"""
        data_wine = pd.read_csv(self.name_path + 'winequality-red.csv')
        X_wine = np.array(data_wine.iloc[:,:-1])

        Y_wine = data_wine.iloc[:,-1]
        # Y_wine=Y_wine[:,None]
        Y_wine = np.where(Y_wine=='g',1.,-1.)
        #normaliza
        ss = StandardScaler()
        X_wine = ss.fit_transform(X_wine)

        return X_wine, Y_wine, data_wine
    

    def spirals(self):
        """## SPIRALs
        """

        data_spiral = pd.read_csv(self.name_path + 'spirals.csv')  

        Y_spiral = data_spiral['classes']
        Y_spiral = np.where( Y_spiral == 1.0, 1.0, -1.0) 

        g = len(data_spiral[data_spiral['classes']==1])
        a = len(data_spiral[data_spiral['classes']==2])
        # sns.countplot(data_spiral['classes'])

        data_spiral = data_spiral.drop(columns=['classes'])
        X_spiral = np.array(data_spiral.values)
        return X_spiral, Y_spiral, data_spiral

    def spirals_t(self):
        """## SPIRALs
        """

        data_spiral = pd.read_csv(self.name_path + 'spiral.csv')  

        Y_spiral = data_spiral['classes']
        Y_spiral = np.where( Y_spiral == 1.0, 1.0, -1.0) 

        g = len(data_spiral[data_spiral['classes']==1])
        a = len(data_spiral[data_spiral['classes']==2])
        # sns.countplot(data_spiral['classes'])

        data_spiral = data_spiral.drop(columns=['classes'])
        X_spiral = np.array(data_spiral.values)

        
        # data_spiral = np.hstack((X_spiral,Y_spiral[:,None]))
        


        # ss = StandardScaler()
        # X_spiral = ss.fit_transform(X_spiral)
        # X_spiral.shape
        # print(g/(a+g)*100, a/(a+g)*100, a+g, X_spiral.shape, Y_spiral.shape)
        return X_spiral, Y_spiral, data_spiral
    
    def spirals_n(self):
        data_spiral_n = pd.read_csv(self.name_path + 'mlp_output.csv')
        data_spiral_n = data_spiral_n.values
        data_spiral_n.shape
        X_spiral_n = data_spiral_n[:,0:19]
        Y_spiral_n = np.array(data_spiral_n[:,20])
        Y_spiral_n = np.where( Y_spiral_n == 1.0, 1.0, -1.0) 

        return X_spiral_n, Y_spiral_n, data_spiral_n

    def normal(self):
        """## NORMAL"""

        data_normal = pd.read_csv(self.name_path + 'normal2.csv')  

        Y_normal = data_normal['classes']
        Y_normal = np.where( Y_normal == 1.0, 1.0, -1.0) 
        data_normal = data_normal.drop(columns=['classes'])
        X_normal = data_normal.values

        data_normal = np.hstack((X_normal,Y_normal[:,None]))
        return X_normal, Y_normal, data_normal
    
    def sonar(self):
        data_sonar = pd.read_csv(self.name_path + 'sonar.csv', header = None)
        data_sonar = data_sonar.values
        X_sonar = data_sonar[:,0:60].astype(float)
        Y_sonar = np.array(data_sonar[:,60])
        Y_sonar = np.where(Y_sonar=='R',-1.,1.)
        # Y_sonar=Y_sonar[:,None]
        #normaliza
        ss = StandardScaler()
        X_sonar = ss.fit_transform(X_sonar)
        X_sonar.shape

        # g = len(data_sonar[data_sonar[:,60]=='R'])
        # a = len(Y_sonar) - g
        # print(g/(a+g)*100, a/(a+g)*100, a+g)

        # len(data_sonar[data_sonar[:,60]=='R'])
        return X_sonar, Y_sonar, data_sonar
    
    def handwritten(self):
        """## Recognition of Handwritten Digits
https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
"""

        data_h = pd.read_csv(self.name_path + 'penbased-5an-nn.csv')
        X_rhd = data_h.iloc[:,:-1]
        Y_rhd = data_h.iloc[:,-1]

        Y_rhd = np.where(Y_rhd<=4,1.0,-1.0)
        # print(Y_rhd)
        # # Y_rhd=Y_rhd[:,None]
        # #normaliza
        ss = StandardScaler()
        X_rhd = ss.fit_transform(X_rhd)
        X_rhd.shape

        # g = len(data_h[data_h.iloc[:,-1]<=4])
        # a = len(Y_rhd) - g
        # print(g/(a+g)*100, a/(a+g)*100, a+g, X_rhd.shape)

        # sns.countplot(Y_rhd)
        return X_rhd, Y_rhd, data_h
    
    def sinc(self):
        Ntreino=250
        Nteste = 50
        X_sinc = np.linspace(-15,15, Ntreino+Nteste)
        X_sinc= X_sinc[:, None]

        ruido = 0.05*np.random.randn(len(X_sinc))
        ruido = ruido[:,None]
        Y_sinc = np.sinc(1/np.pi*X_sinc) + ruido

        return X_sinc, Y_sinc, []
    
    def ionosphere(self):
        """## Ionosphere
        https://archive.ics.uci.edu/ml/datasets/ionosphere

        Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266.

        -- All 34 are continuous (1 not used)
        -- The 35th attribute is either "good" or "bad" according to the definition summarized above. This is a binary classification task.

        """
        data = pd.read_csv(self.name_path + 'ionosphere_data.csv')
        data.drop(columns=['column_b'], inplace=True)
        data['column_a'] = data.column_a.astype('float64')
        X_ion = np.array(data.iloc[:,:-1])

        Y_ion = data.iloc[:,-1]
        Y_ion = np.where(Y_ion=='g',1.0,-1.0)
        # Y_ion=Y_ion[:,None]

        #normaliza
        ss = StandardScaler()
        X_ion[:,1:] = ss.fit_transform(X_ion[:,1:])
        # X_ion.shape
        # ax = sns.countplot(x='column_ai',data=data)
        # g = len(data[data['column_ai']=='g'])
        # a = len(data[data['column_ai']=='b'])
        # print(g/(a+g)*100, a/(a+g)*100, a+g)
        return X_ion, Y_ion, data
    
    def automobile(self):
        data_auto = pd.read_csv(self.name_path + 'automobile.csv')
        total = data_auto.isnull().sum().sort_values(ascending=False)
        percent = ((data_auto.isnull().sum())*100)/data_auto.isnull().count().sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
        missing_data.head(40)

        a=data_auto[data_auto['normalized-losses']!='?']
        b=(a['normalized-losses'].astype(int)).mean()
        data_auto['normalized-losses']=data_auto['normalized-losses'].replace('?',b).astype(int)

        a=data_auto['num-of-doors'].map({'two':2,'four':4,'?':4})
        data_auto['num-of-doors']=a

        a=data_auto[data_auto['price']!='?']
        b=(a['price'].astype(int)).mean()
        data_auto['price']=data_auto['price'].replace('?',b).astype(int)

        a=data_auto[data_auto['horsepower']!='?']
        b=(a['horsepower'].astype(int)).mean()
        data_auto['horsepower']=data_auto['horsepower'].replace('?',b).astype(int)

        a=data_auto[data_auto['bore']!='?']
        b=(a['bore'].astype(float)).mean()
        data_auto['bore']=data_auto['bore'].replace('?',b).astype(float)

        a=data_auto[data_auto['stroke']!='?']
        b=(a['stroke'].astype(float)).mean()
        data_auto['stroke']=data_auto['stroke'].replace('?',b).astype(float)

        a=data_auto[data_auto['peak-rpm']!='?']
        b=(a['peak-rpm'].astype(float)).mean()
        data_auto['peak-rpm']=data_auto['peak-rpm'].replace('?',b).astype(float)

        a=data_auto['num-of-cylinders'].map({'four':4,'five':5,'six':6,'?':4})
        data_auto['num-of-doors']=a

        a=data_auto['num-of-cylinders'].map({'four':4,'five':5,'six':6,'?':4})
        data_auto['num-of-doors']=a

        dictOfWords = { data_auto['body-style'].unique()[i] : i for i in range(0, len(data_auto['body-style'].unique()) ) }
        data_auto['body-style'] = data_auto['body-style'].map(dictOfWords)

        dictOfWords = { data_auto['make'].unique()[i] : i for i in range(0, len(data_auto['make'].unique()) ) }
        data_auto['make'] = data_auto['make'].map(dictOfWords)

        dictOfWords = { data_auto['fuel-type'].unique()[i] : i for i in range(0, len(data_auto['fuel-type'].unique()) ) }
        data_auto['fuel-type'] = data_auto['fuel-type'].map(dictOfWords)

        dictOfWords = { data_auto['aspiration'].unique()[i] : i for i in range(0, len(data_auto['aspiration'].unique()) ) }
        data_auto['aspiration'] = data_auto['aspiration'].map(dictOfWords)


        dictOfWords = { data_auto['drive-wheels'].unique()[i] : i for i in range(0, len(data_auto['drive-wheels'].unique()) ) }
        data_auto['drive-wheels'] = data_auto['drive-wheels'].map(dictOfWords)

        dictOfWords = { data_auto['engine-location'].unique()[i] : i for i in range(0, len(data_auto['engine-location'].unique()) ) }
        data_auto['engine-location'] = data_auto['engine-location'].map(dictOfWords)

        dictOfWords = { data_auto['engine-type'].unique()[i] : i for i in range(0, len(data_auto['engine-type'].unique()) ) }
        data_auto['engine-type'] = data_auto['engine-type'].map(dictOfWords)

        dictOfWords = { data_auto['num-of-cylinders'].unique()[i] : i for i in range(0, len(data_auto['num-of-cylinders'].unique()) ) }
        data_auto['num-of-cylinders'] = data_auto['num-of-cylinders'].map(dictOfWords)

        dictOfWords = { data_auto['fuel-system'].unique()[i] : i for i in range(0, len(data_auto['fuel-system'].unique()) ) }
        data_auto['fuel-system'] = data_auto['fuel-system'].map(dictOfWords)


        data_auto["price"]=data_auto["price"].astype(float)
        # data_auto = data_auto.drop(columns=['symboling','normalized-losses'])

        Y_auto = np.array(data_auto["price"]).reshape(-1, 1)
        data_auto.drop(columns=["price"],inplace=True)
        X_auto = np.array(data_auto)
        # #normaliza
        ss = StandardScaler()
        X_auto = ss.fit_transform(X_auto)

        # ss2 = MinMaxScaler()
        # Y_auto = ss2.fit_transform(Y_auto)
        
        return X_auto, Y_auto, data_auto
    
    
    def iris(self):
        """DATASET : IRIS"""        

        iris = load_iris()
        X_iris = iris.data
        Y_iris = iris.target
        Y_iris = np.where(Y_iris==1,-1.,1.)
        # Y_iris=Y_iris[:,None]
        #normaliza
        ss = StandardScaler()
        X_iris = ss.fit_transform(X_iris)
        # X_iris.shape
        return X_iris, Y_iris, iris
    
    def breastcancer(self):
        df = pd.read_csv('./dataset/breastcancer.csv')
        df = df.drop(['Unnamed: 32', 'id'], axis=1)
        df['diagnosis']= df['diagnosis'].replace('M', 1)
        df['diagnosis']= df['diagnosis'].replace('B', -1)
        df = df[['diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean',
            'radius_worst', 'perimeter_worst', 'area_worst', 'compactness_worst',
            'concavity_worst', 'concave points_worst']]
        x = df.drop(['diagnosis'], axis=1).to_numpy()
        stdScaler_data = StandardScaler()
        X = stdScaler_data.fit_transform(x) 
        Y = df['diagnosis'].to_numpy() 
        

        return X, Y, df   
        # data = load_breast_cancer()
        # X, Y = data["data"], data["target"]

        # stdScaler_data = StandardScaler()
        # X = stdScaler_data.fit_transform(X) 

        # return X, Y, data   
        # dataset = load_breast_cancer()
        # X = sklearn.preprocessing.normalize(dataset.data)
        # y = dataset.target
        # # y = np.reshape((y.shape[0],1),y)
        # y = np.where(y==0,-1,1)
        
        # # data_4 = np.hstack((X, y)) 
        # return X, y, dataset
    



    def load(self,name_dataset): 
        self.datasets[name_dataset]  
        print(name_dataset)
        return  self.datasets[name_dataset]
        



# ds = DatasetsLoad()
# X_xor, Y_xor, data_xor = ds.load('xor')
# print(X_xor.shape)
