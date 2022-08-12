from data_clean import DataSet
from sklearn.model_selection import StratifiedKFold

class Kfold(object):
    # data = DataSet()
    def __init__(self,data):
        self.data = data
        
    def kfold(self,n_splits = 5):
        kf = StratifiedKFold(n_splits)
        return kf
    
#     # def train_val_test_split(self,kf,X,y):
#     #     for train_idx,test_idx in kf.split(X,y):
#     #         print(train_idx)
#     #         X_train = X.loc[train_idx,X.columns.to_list()]
#     #         y_train = y[train_idx]
            
#     #         X_val = X.loc[test_idx,X.columns.to_list()]
#     #         y_val = y[test_idx]
            
#     #     return X_train,y_train,X_val,y_val

# class Model(object):
#     def()