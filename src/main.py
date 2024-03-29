from xmlrpc.client import Boolean
from data_clean import DataSet
from kfold import Kfold
from model import Model
import argparse
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--tree',type=Boolean,default=False,help="Tree based Algo or Non tree based algo",required=True)
    parser.add_argument('-a','--algo',choices=['LR','NB','SVM','KNN','DT','RF','XGB'],help='Choose which ML algo to run',default='RF',required=True)
    parser.add_argument('-ns','--nsplits',type=int,default=5,help='number of splits in kfold')
    results = parser.parse_args()
    data = DataSet()
    data.load_csv()
    int_col,str_col = data.continuous_categorical()
    data.encoding()
    print(data.details())
    
    kf = Kfold(data)
    fold = kf.kfold()
    X,y = data.data_target()
    
    m = Model(data)
    
    if results.tree == True:
        if results.algo == 'DT':
            metric = m.train_predict(X,y,fold,DecisionTreeClassifier(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")
        elif results.algo == 'RF':
            metric = m.train_predict(X,y,fold,RandomForestClassifier(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")
        elif results.algo == 'XGB':
            metric = m.train_predict(X,y,fold,XGBClassifier(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")
    else:
        if results.algo == 'LR':
            metric = m.train_predict(X,y,fold,LogisticRegression(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")
        elif results.algo == 'NB':
            metric = m.train_predict(X,y,fold,GaussianNB(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")
        elif results.algo == 'SVM':
            metric = m.train_predict(X,y,fold,SVC(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")
        elif results.algo == 'KNN':
            metric = m.train_predict(X,y,fold,KNeighborsClassifier(),tree=results.tree)
            # print(f"Best metric for {results.algo} Algorithm is -------> {max(metric)}")  
    
    
    
    

    