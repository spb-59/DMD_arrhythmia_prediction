
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import logging as lg
import time
import os
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import scipy.stats as stats

from sklearn.neighbors import KNeighborsClassifier


'''
This script was used to run the classification model on various different extracted features all at once.
The models are the same as seen in the notebooks.
'''


def cleanData(df:pd.DataFrame,i:int):
    df=df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df['Label']=i
    return df
    
def preprocess_data(X):

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()  

  
    for column in X.select_dtypes(include=['object']).columns:

        if not X[column].apply(lambda x: isinstance(x, str)).all():
            X[column] = X[column].astype(str)

        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    
    return X

def significanceTest(SR,COND):
    significant=[]



    for column in SR.columns:
        if column in COND.columns and column!="Label":  
            srVal = SR[column]
            condVal = COND[column] 
            

            _, p_value = stats.mannwhitneyu(srVal, condVal, alternative='two-sided')
            if p_value<0.05:
                significant.append(column)
    # print(significant)
    return significant

def getSplit(files,num=2,scalar=False):

    
    datasets=cleanData(pd.read_csv(f'features{num}/SR.csv',low_memory=False),0).iloc[:1000]
    SR=datasets.copy()
    
    label=1
    for filename in os.listdir(f'features{num}'):
    
        if filename.endswith('.csv') and filename != 'SR.csv' and not "Unknown" in filename and filename in files:
        
            df = cleanData(pd.read_csv(os.path.join(f'features{num}', filename),low_memory=False), label).iloc[:1000]
            sig=significanceTest(SR,df)
            sig.append('Label')
            label+=1
            datasets=pd.concat([datasets[sig],df[sig]])
    
    X = datasets.drop(["Label"], axis =1)
   
    
    Y = datasets["Label"]




    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=42)
    if scalar:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 
    return X_train,Y_train, X_test,  Y_test

def RF(X_tr, Y_tr, X_te, Y_te):

    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],  
        'max_depth': range(1, 20),  
        'criterion': ['gini', 'entropy'] 
    }


    optimal_params = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=10, 
        scoring='accuracy',
        verbose=0,
        n_jobs=-1
    )


    optimal_params.fit(X_tr, Y_tr)
    print("Best parameters found: ", optimal_params.best_params_)


    criterion = optimal_params.best_params_['criterion']
    max_depth = optimal_params.best_params_['max_depth']
    n_estimators = optimal_params.best_params_['n_estimators']


    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    rf_model.fit(X_tr, Y_tr)


    rf_pred = rf_model.predict(X_te)


    plot=ConfusionMatrixDisplay.from_estimator(estimator=rf_model, X=X_te, y=Y_te)
    plot.ax_.set_title('Confusion Matrix')
    plt.show()
    print("Best Cross-Validation Score:",optimal_params.best_score_)
    print("Classification Report: Random Forest")
    print(classification_report(Y_te, rf_pred, digits=2),roc_auc_score(Y_te,rf_pred))
    with open('results.txt', 'a') as f:
        f.write(classification_report(Y_te, rf_pred, digits=2))

def XGBoost(X_tr, Y_tr, X_te, Y_te):
    # X_tr = preprocess_data(X_tr)
    # X_te = preprocess_data(X_te)


    class_weight = compute_class_weight('balanced', classes=np.unique(Y_tr), y=Y_tr)
    lg.info("Starting hyperparameter optimization for XGBoost")


    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    # Create the model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=42
    )


    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',  
        cv=10,  
        verbose=1,
        n_jobs=-1  
    )

  
    grid_search.fit(X_tr, Y_tr)

   
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    lg.info(f"Best parameters found: {best_params}")
    lg.info(f"Best cross-validated F1 score: {best_score:.4f}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

    print("Best Cross-Validation Score:", grid_search.best_score_)
    xgb_pred = best_model.predict(X_te)

    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: XGBoost",roc_auc_score(Y_te,xgb_pred))
    report = classification_report(Y_te, xgb_pred, digits=2)
    print(report)

def SVM(X_tr, Y_tr, X_te, Y_te):
    param_grid = {
            'C': [0.1, 1, 10, 100],  
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  
            'kernel': ['linear', 'rbf'],  
            'class_weight': ['balanced', None]  
        }


    model = SVC(random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                            cv=5, scoring='accuracy', n_jobs=-1)


    grid_search.fit(X_tr, Y_tr)
    

    print("Best parameters found: ", grid_search.best_params_)

 
    model = grid_search.best_estimator_
    Y_pred = model.predict(X_te)

 
    print("Classification Report:\n", classification_report(Y_te, Y_pred),roc_auc_score(Y_te,Y_pred))
    print("Best Cross-Validation Score:", grid_search.best_score_)
 

    plot = ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_te, y=Y_te)
    plot.ax_.set_title('Confusion Matrix')
    plt.show()



def KNN(X_tr, Y_tr, X_te, Y_te):


    model = KNeighborsClassifier()
    

    param_grid = {
        'n_neighbors': [x for x in range(3,9)],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy')
    

    grid_search.fit(X_tr, Y_tr)
    

    best_model = grid_search.best_estimator_
    

    Y_pred = best_model.predict(X_te)
    
    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)
    print("Classification Report:\n", classification_report(Y_te, Y_pred),roc_auc_score(Y_te,Y_pred))


for i in [5,6,8,2]:
    print("AFIB>>>>>>>>>>",i)
    X_tr, Y_tr, X_te, Y_te=getSplit(['AFIB.csv'],i,True)
    RF(X_tr, Y_tr, X_te, Y_te)
    SVM(X_tr, Y_tr, X_te, Y_te)
    KNN(X_tr, Y_tr, X_te, Y_te)
    XGBoost(X_tr, Y_tr, X_te, Y_te)
    print('AF>>>>>>>>',i)
    X_tr, Y_tr, X_te, Y_te=getSplit(['AF.csv'],i,True)
    RF(X_tr, Y_tr, X_te, Y_te)
    SVM(X_tr, Y_tr, X_te, Y_te)
    KNN(X_tr, Y_tr, X_te, Y_te)
    XGBoost(X_tr, Y_tr, X_te, Y_te)

print('>>>>>>>>MIMI')
X_tr, Y_tr, X_te, Y_te=getSplit(['MI.csv'],4,True)
RF(X_tr, Y_tr, X_te, Y_te)
SVM(X_tr, Y_tr, X_te, Y_te)
KNN(X_tr, Y_tr, X_te, Y_te)
XGBoost(X_tr, Y_tr, X_te, Y_te)

for i in [10,11]:
    print("MI>>>>>>>>>>",i)
    X_tr, Y_tr, X_te, Y_te=getSplit(['MI.csv'],i,True)
    RF(X_tr, Y_tr, X_te, Y_te)
    SVM(X_tr, Y_tr, X_te, Y_te)
    KNN(X_tr, Y_tr, X_te, Y_te)
    XGBoost(X_tr, Y_tr, X_te, Y_te)
    print("AFIB>>>>>>>>>>",i)
    X_tr, Y_tr, X_te, Y_te=getSplit(['AFIB.csv'],i,True)
    RF(X_tr, Y_tr, X_te, Y_te)
    SVM(X_tr, Y_tr, X_te, Y_te)
    KNN(X_tr, Y_tr, X_te, Y_te)
    XGBoost(X_tr, Y_tr, X_te, Y_te)


