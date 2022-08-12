from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score



class Model(object):
    def __init__(self,data):
        self.data  = data
    
    def model(self,algo=RandomForestClassifier()):
        return algo
    
    def fit(self,Xtrain,ytrain,m):
        m.fit(Xtrain,ytrain)
    
    def predict(self,m,Xtest):
        y_pred = m.predict(Xtest)
        return y_pred
    
    
    