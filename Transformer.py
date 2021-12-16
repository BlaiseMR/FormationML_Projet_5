import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    # Transformer to perform an ACP trnsformation
    def __init__(self, feature_names=None, n_comp=None):
        if feature_names is not None:
            self.feature_names = feature_names
        else :
            self.feature_names = None
            
        if n_comp is not None:
            self.n_comp = n_comp
        else :
            self.n_comp = None
            
        self.Names = []
        
        
    def fit(self, X, Y=None):
        self.X_copy = X.copy()
        if self.feature_names is None :
            self.feature_names = self.X_copy.columns
        else :
            self.X_copy = self.X_copy[self.X_copy.columns.intersection(self.feature_names)]
            
        if self.n_comp is None:
            self.n_comp = len(self.feature_names)
       
        self.X_copy = self.X_copy.fillna(self.X_copy.mean())
        self.pca = decomposition.PCA(n_components=self.n_comp)
        self.pca.fit(self.X_copy)
        return self
    
    def transform(self, X, Y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy.columns.intersection(self.feature_names)]
        X_copy = X_copy.fillna(X_copy.mean())
        
        values = X_copy.values
        features = X_copy.columns
        
        for i in range(self.n_comp):
            self.Names.append('F%d' % i)

        X_projected = self.pca.transform(values)

        result = pd.DataFrame(X_projected, columns = self.Names)
        for feat in X.columns: 
            if feat not in self.feature_names:
                result[feat] = X[feat]
        result.replace([-np.inf, np.inf], np.nan, inplace=True)
        result = result.fillna(0)
        return result

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X, Y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy.columns.intersection(self.feature_names)]
        X_copy = X_copy.fillna(X_copy.mean())
        
        values = X_copy.values
        features = X_copy.columns

        X_scaled = self.pca.inverse_transform(values)
        
        result = pd.DataFrame(X_scaled, columns = self.feature_names)
        for feat in X.columns: 
            if feat not in self.Names:
                result[feat] = X[feat]
        
        result.replace([-np.inf, np.inf], np.nan, inplace=True)
        result = result.fillna(0)
        
        return result

class MyStandradScaler(BaseEstimator, TransformerMixin):
    # Transformer to perform a StandardScaler
    def __init__(self, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        else :
            self.feature_names = None
        self.scaler = preprocessing.StandardScaler()
#         print('Init :', self.feature_names)
        
    def fit(self, X, Y=None):
        self.X_copy = X.copy()
        if self.feature_names is None :
            self.feature_names = self.X_copy.columns
        else :
            self.X_copy = self.X_copy[self.X_copy.columns.intersection(self.feature_names)]
            
        self.X_copy = self.X_copy.fillna(self.X_copy.mean())
        self.scaler.fit(self.X_copy)
        return self
    
    def transform(self, X, Y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy.columns.intersection(self.feature_names)]
        X_copy = X_copy.fillna(X_copy.mean())
        
        Names = []
        for i, name in enumerate(self.feature_names):
            Names.append(name)

        X_scaled = self.scaler.transform(X_copy)
        
#         print('Transform :', self.feature_names)
        result = pd.DataFrame(X_scaled, columns = Names)
        for feat in X.columns: 
            if feat not in self.feature_names:
                result[feat] = X[feat]
        result.replace([-np.inf, np.inf], np.nan, inplace=True)
        result = result.fillna(0)
        
        return result
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X, Y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy.columns.intersection(self.feature_names)]
        X_copy = X_copy.fillna(X_copy.mean())

        Names = []
        for i, name in enumerate(self.feature_names):
            Names.append(name)

        X_scaled = self.scaler.inverse_transform(X_copy)

        #         print('Transform :', self.feature_names)
        result = pd.DataFrame(X_scaled, columns = Names)
        for feat in X.columns: 
            if feat not in self.feature_names:
                result[feat] = X[feat]
                
        result.replace([-np.inf, np.inf], np.nan, inplace=True)
        result = result.fillna(0)
        return result
    
class MyNormScaler(BaseEstimator, TransformerMixin):
    # Transformer to perform a normalisation
    def __init__(self, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = None
        
        self.scaler = preprocessing.MinMaxScaler()
        
    def fit(self, X, Y=None):
        self.X_copy = X.copy()
        if self.feature_names is None :
            self.feature_names = self.X_copy.columns
        else :
            self.X_copy = self.X_copy[self.X_copy.columns.intersection(self.feature_names)]
            
        self.X_copy = self.X_copy.fillna(self.X_copy.mean())
        self.scaler.fit(self.X_copy)
        return self
    
    def transform(self, X, Y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy.columns.intersection(self.feature_names)]
        X_copy = X_copy.fillna(X_copy.mean())
        
        Names = []
        for i, name in enumerate(self.feature_names):
            Names.append(name)

        X_scaled = self.scaler.transform(X_copy)
        
#         print('Transform :', self.feature_names)
        result = pd.DataFrame(X_scaled, columns = Names)
        for feat in X.columns: 
            if feat not in self.feature_names:
                result[feat] = X[feat]
        result.replace([-np.inf, np.inf], np.nan, inplace=True)
        result = result.fillna(0)
        
        return result
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X, Y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy.columns.intersection(self.feature_names)]
        X_copy = X_copy.fillna(X_copy.mean())
        
        Names = []
        for i, name in enumerate(self.feature_names):
            Names.append(name)

        X_scaled = self.scaler.inverse_transform(X_copy)
        
#         print('Transform :', self.feature_names)
        result = pd.DataFrame(X_scaled, columns = Names)
        for feat in X.columns: 
            if feat not in self.feature_names:
                result[feat] = X[feat]
        
        result.replace([-np.inf, np.inf], np.nan, inplace=True)
        result = result.fillna(0)
        
        return result