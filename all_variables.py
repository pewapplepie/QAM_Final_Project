import pandas as pd # pandas is used to load and manipulate data and for One-Hot Encoding
import numpy as np # data manipulation
import os
import matplotlib.pyplot as plt # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample # downsample the dataset
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn import preprocessing # scale and center data
from sklearn.model_selection import GridSearchCV # this will do cross validation
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.neural_network import MLPRegressor
import datetime
from pandas.tseries.offsets import *
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error

data_folder = 'C:/Users/Damian/Desktop/QAM/final/'
macro_data=pd.read_csv(data_folder + 'Macro_Result.csv', index_col=0)
macro_data.rename(columns={'ep': 'earning_price'}, inplace=True)

stk_cahr_data=pd.read_csv(data_folder+'neural_net_full.csv', index_col=0)
stk_cahr_data.drop(columns=['TICKER','month','year'], inplace=True)
cit_cols = stk_cahr_data.iloc[:,3:97].columns

macro_data['date'] = pd.to_datetime(macro_data['yyyymm'].astype(str), format='%Y%m', errors='ignore') + MonthEnd(1)
macro_data['Year'] = macro_data['date'].dt.year
macro_data['Month'] = macro_data['date'].dt.month

stk_cahr_data['DATE'] = pd.to_datetime(stk_cahr_data['DATE'], format='%Y-%m-%d', errors='ignore')
stk_cahr_data['Year'] = stk_cahr_data['DATE'].dt.year
stk_cahr_data['Month'] = stk_cahr_data['DATE'].dt.month
all_data = stk_cahr_data.merge(macro_data, on=['Year', 'Month'], how='left').copy()
all_data['Const'] = 1
all_data.drop(columns=['Year','Month','yyyymm','date'], inplace=True)

xt_cols = ['dp', 'earning_price', 'tms', 'dfy', 'b/m', 'ntis', 'tbl', 'svar', 'Const']
Pc = len(cit_cols)
Px = len(xt_cols)

print(f'total covariates should be: Pxt * Pcit = {Pc} * {Px} = {Pc * Px}')


all_data['sic2']=all_data['sic2'].astype(object)
#Removed one level as we have Const
industry = pd.get_dummies(all_data.sic2).iloc[:,:-1]
# all_data = all_data.join(pd.get_dummies(all_data.sic2).iloc[:,:-1])
all_data.to_csv(data_folder + 'macro_stock_dummies.csv')
all_data.drop(columns=['sic2'], inplace=True)

industry.columns = [str(col) + '_i' for col in industry.columns]

def kro(row):
    Cit = row[cit_cols]
    Xt = row[xt_cols]
    zit = np.kron(Xt, Cit)
    return zit

Zit = pd.DataFrame(all_data.apply(kro, axis=1).to_list())


Zit.insert (0, "permno", all_data['permno'])
Zit.insert (1, "DATE", all_data['DATE'])
Zit.insert (2, "monthly_ret", all_data['monthly_ret'])
Zit.sort_values(by=['DATE'])
Zit

kronercker_outcome = Zit.copy()
kronercker_outcome = kronercker_outcome.fillna(0)
kronercker_outcome['Year'] = kronercker_outcome['DATE'].dt.year
kronercker_outcome['Month'] = kronercker_outcome['DATE'].dt.month
#Choose 2000
train_df = kronercker_outcome[kronercker_outcome['Year']<2000]
test_df = kronercker_outcome[kronercker_outcome['Year'] >= 2000]

X_train = train_df.iloc[:,3:849]
y_train = train_df['monthly_ret']
X_test = test_df.iloc[:,3:849]
y_test = test_df['monthly_ret']


# scale X variables to mean zero and unit variance
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled).join(industry).to_numpy()
X_test_scaled = pd.DataFrame(X_test_scaled).join(industry).to_numpy()

NN3 = MLPRegressor(solver = 'lbfgs', alpha = 1, activation='tanh',
                      hidden_layer_sizes = (32,16, 8,), max_iter = 100000, random_state = 1)

NN3.fit(X_train_scaled,y_train)

# R2 of neural net fit, in sample
R2_NN3_insample = NN3.score(X_train_scaled, y_train, sample_weight=None)

# R2 of neural net fit, out of sample
R2_NN3_outsample = NN3.score(X_test_scaled, y_test, sample_weight=None)

print(
    f'R2 insample : {R2_NN3_insample}\n'
    f'R2 outsample : {R2_NN3_outsample}'
    )