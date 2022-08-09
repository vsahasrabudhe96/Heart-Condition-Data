from turtle import pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

class DataSet(object):
    data_path = "../data/"
    def __init__(self,data_path=None):
        
        self.data_path = data_path
        
        if not self.data_path:
            if os.path.exists(DataSet.data_path):
                self.data_path = DataSet.data_path
            else:
                print(f"The path {DataSet.data_path} does not exist")
    
    
    def load_csv(self):
        self.data = pd.read_csv(os.path.join(self.data_path,'heart.csv'))
    
    def print_df(self,h=5):
        print("Printing the Data head -------")
        print(self.data.head(h))
        print("Printing the Data tail -------")
        print(self.data.tail(h))
    
    def continuous_categorical(self):
        int_col = [c for c in self.data.columns if self.data[c].dtype == 'int64' or self.data[c].dtype == 'float64']
        str_col = [c for c in self.data.columns if c not in int_col]
        return int_col,str_col
    
    def encoding(self,tree=True):
        int_col, str_col = self.continuous_categorical()
        if tree == True:
            self.data = self.data.apply(LabelEncoder().fit_transform)
        else:
            self.data = pd.get_dummies(self.data,columns=str_col,drop_first=True)
                
        
        
                