from xmlrpc.client import Boolean
from data_clean import DataSet
from kfold import Kfold
from model import Model
import argparse
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score

if __name__ == "__main__":
    data = DataSet()
    data.load_csv()
    int_col,str_col = data.continuous_categorical()
    data.encoding()
    print(data.details())
    
    kf = Kfold(data)
    fold = kf.kfold()
    X,y = data.data_target()
    
    m = Model(data)
    acc_log = []
    for fold, (train_idx,test_idx) in enumerate(fold.split(X=X,y=y.values)):
        X_train = X.loc[train_idx,X.columns.to_list()]
        y_train = y[train_idx]
            
        X_val = X.loc[test_idx,X.columns.to_list()]
        y_val = y[test_idx]
        
        model = m.model()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_val)
        print(f"The fold is : {fold} : ")
        print(classification_report(y_val,y_pred))
        acc = roc_auc_score(y_val,y_pred)
        acc_log.append(acc)
        
        print(f"The accuracy for Fold {fold+1} : {acc}")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--tree',type=Boolean,default=False,help="Tree based Algo or Non tree based algo")
    parser.add_argument('-a','--algo',choices=['LR','NB','SVM','KNN','DT','RF','XGB'],help='Choose which ML algo to run')
    parser.add_argument('-ns','--nsplits',type=int,default=5,help='number of splits in kfold')
    

    