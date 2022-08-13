
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.preprocessing import MinMaxScaler




class Model(object):
    def __init__(self,data):
        self.data  = data
    
    def fit(self,Xtrain,ytrain,m):
        m.fit(Xtrain,ytrain)
    
    def predict(self,m,Xtest):
        y_pred = m.predict(Xtest)
        return y_pred
    
    def train_predict(self,X,y,kf,model,tree =True):
        acc_log = []
        for fold, (train_idx,test_idx) in enumerate(kf.split(X=X,y=y.values)):
            X_train = X.loc[train_idx,X.columns.to_list()]
            y_train = y[train_idx]
                
            X_val = X.loc[test_idx,X.columns.to_list()]
            y_val = y[test_idx]
            if tree == False:
                print("--------Performing MinMaxScaling-----------")
                ro_scaler=MinMaxScaler()
                X_train=ro_scaler.fit_transform(X_train)
                X_val=ro_scaler.transform(X_val)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_val)
                print(f"The fold is : {fold} : ")
                print(classification_report(y_val,y_pred))
                acc = roc_auc_score(y_val,y_pred)
                acc_log.append(acc)
                print(f"The accuracy for Fold {fold+1} : {acc}")
            else:
                model.fit(X_train,y_train)
                y_pred = model.predict(X_val)
                print(f"The fold is : {fold} : ")
                print(classification_report(y_val,y_pred))
                acc = roc_auc_score(y_val,y_pred)
                acc_log.append(acc)
                print(f"The accuracy for Fold {fold+1} : {acc}")
                
        
        return acc_log