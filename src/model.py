from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



class Model(object):
    def __init__(self,data):
        self.data  = data
    
    def model(self):
        m = RandomForestClassifier()
        return m
    
    def fit(self,Xtrain,ytrain):
        
        pass