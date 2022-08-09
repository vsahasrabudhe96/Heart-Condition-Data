from turtle import pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
    
    
        
        
                