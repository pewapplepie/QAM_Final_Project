import os
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import *


data_folder = '/Users/jeffreychen/UCLA_HWDATA/QAM_DATA/final_data/'  

data = pd.read_csv(data_folder + 'datashare.csv') 

permno = pd.read_csv('permno.csv', index = False)

tech50 = data[data['permno'].isin(permno['permno'])]

tech50['date'] = pd.to_datetime(tech50['DATE'], format='%Y%m%d',errors='ignore')
tech50['month'] = pd.DatetimeIndex(tech50['date']).month
tech50['year'] = pd.DatetimeIndex(tech50['date']).year

permno['month'] = pd.DatetimeIndex(permno['date']).month
permno['year'] = pd.DatetimeIndex(permno['date']).year

return_and_vars = pd.merge(tech50,permno[['permno','TICKER','month','year','Monthly Stock Return']],how='inner', on=['permno','year','month'])
return_and_vars['DATE'] = return_and_vars['date']
first = return_and_vars['Monthly Stock Return']
return_and_vars.drop(labels=['date','Monthly Stock Return'], axis=1,inplace = True)
return_and_vars.insert(2,'monthly_ret',first)

variable_top20 = return_and_vars[['permno','DATE','monthly_ret', 'mom1m','maxret','chmom','mom6m','mom12m','indmom','mom36m',\
                                  'mvel1','dolvol','turn','baspread','ill','std_turn','zerotrade',\
                                  'retvol','idiovol','beta',\
                                  'nincr','sp','securedind']]

variable_top20.to_csv(data_folder + 'neural_net20.csv')

return_and_vars.to_csv(data_folder + 'neural_net_full.csv')