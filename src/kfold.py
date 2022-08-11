from data_clean import DataSet
from sklearn.model_selection import StratifiedKFold

class Kfold(object):
    # data = DataSet()
    def __init__(self,data):
        self.data = data
        
    def kfold(self,n_splits = 5):
        kf = StratifiedKFold(n_splits)
        return kf
    
    
        
        