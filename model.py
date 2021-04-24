import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder 
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import xgboost as xgb1
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pickle

train_data = pd.read_csv('train.csv')
test_data  = pd.read_csv('test.csv')

test_data['loss'] = np.nan
combined_data = pd.concat([train_data, test_data])

#seperating categorical and continuous features
def features(df):
    categorical_features=[]
    continuous_features=[]
    for i in df.columns:
        if i[:3]=='cat':
            categorical_features.append(i)
        elif i[:4]=='cont':
            continuous_features.append(i)
    return categorical_features,continuous_features

categorical_train_features,continuous_train_features=features(train_data)
categorical_test_features,continuous_test_features=features(test_data)

#https://github.com/Ch-Balaji/AllState-Claims-Prediction/blob/master/Final%20-%20cs-%20modelling.ipynb
def search_feature(x):
    if x in combined_remaining:
        return np.nan
    return 

#Reference https://www.geeksforgeeks.org/python-pandas-factorize/
#Reference https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe

for i in categorical_train_features:
    if train_data[i].nunique() != test_data[i].nunique():
        train_unique_set = set(train_data[i].unique())
        test_unique_set  = set(test_data[i].unique())
        remaining_train  = train_unique_set - test_unique_set
        remaining_test   = test_unique_set - train_unique_set
        
        combined_remaining = remaining_train.union(remaining_test)
        
        combined_data[i] = combined_data[i].apply(lambda x: search_feature(x),1)
    combined_data[i] = pd.factorize(combined_data[i].values,sort = True)[0]

x_train = combined_data[combined_data['loss'].notnull()]
x_test = combined_data[combined_data['loss'].isnull()]
y_train = np.log(x_train['loss']+100)
x_train = x_train.drop(['loss','id'],axis = 1)
x_test  = x_test.drop(['loss','id'],axis = 1)

d_train = xgb1.DMatrix(x_train, label=y_train)
d_test =  xgb1.DMatrix(x_test)

params = {'min_child_weight':3,'eta':0.01,'colsample_bytree':0.9,'max_depth':5,'subsample':0.9,'alpha':100,'gamma':0.0,'seed':1997}
def log_xgboost_eval_mae(pred,d_train):
    labels = d_train.get_label()
    a = mean_absolute_error(np.exp(pred)-100,np.exp(labels)-100)
    return 'mae',a
# Finding best rounds via cross validate xgboost with 50 early stoppings.
res = xgb1.cv(params,d_train, num_boost_round=5500, nfold=5, stratified=False,early_stopping_rounds=50, verbose_eval=500, show_stdv=True, feval=log_xgboost_eval_mae, maximize=False)
model = xgb1.train(params,d_train,int(5499), feval=log_xgboost_eval_mae)

filename = 'XGBOOST_FINAL_SUBMISSION_MODEL.pkl'
pickle.dump(model, open(filename, 'wb'))