from data_clean import DataSet
from kfold import Kfold

if __name__ == "__main__":
    data = DataSet()
    data.load_csv()
    int_col,str_col = data.continuous_categorical()
    data.encoding()
    print(data.detais())
    
    kf = Kfold(data)
    fold = kf.kfold()
    X,y = data.data_target()
    
    
    