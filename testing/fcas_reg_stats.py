# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:10:01 2021

@author: benne
"""

import genOpt as go
import datetime as dt
import sys
import os
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import wget
import zipfile
import shutil
import numpy as np
import random
import copy
import dateutil.relativedelta as rdelta
import h5py
myPath = os.getcwd()

#%% Load dependencies
rrp_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\prices\20210505_NEM_20200701-00-05_20210301-00-00_RRP.csv"
dispatchload_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\dispatchload\20210507_NEM_20200701-00-05_20210301-00-00_DISPATCHLOAD.csv"
fcas_reg_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\fcas_reg\reg_component_5min.csv"

# Read in the files
rrp = pd.read_csv(rrp_file,parse_dates=True,index_col=0)
dispatchload_raw = pd.read_csv(dispatchload_file,parse_dates=True,index_col=0)
fcas_reg_raw = pd.read_csv(fcas_reg_file,parse_dates=True,index_col=0)

# Set all index names to Timestamp
rrp.index.name = 'Timestamp'
dispatchload_raw.index.name = 'Timestamp'
fcas_reg_raw.index.name = 'Timestamp'

# Reset indices
rrp.reset_index(inplace=True)
dispatchload_raw.reset_index(inplace=True)
fcas_reg_raw.reset_index(inplace=True)

#%% Merge generator and load duids in dispatchload
dispatchload = dispatchload_raw.copy()
dispatchload['duid pre'] = dispatchload['DUID'].apply(lambda x: x[:-2])
# dispatchload['duid post'] = dispatchload['DUID'].apply(lambda x: x[-2:])
# dispatchload = dispatchload.dropna(how='all',axis=1)
dispatchload = dispatchload.drop(['INTERVENTION'],axis=1)
# dispatchload.loc[dispatchload['duid post'] == 'L1',['INITIALMW','TOTALCLEARED']] *= -1
dispatchload = dispatchload.groupby(['Timestamp','duid pre']).sum().reset_index()

#%% Map state onto fcas_reg
fcas_reg = fcas_reg_raw.copy()

states = {
    'BALB': 'VIC1',
    'GANNB': 'VIC1',
    'HPR': 'SA1',
    'LBB': 'SA1'
    }

fcas_reg['REGIONID'] = fcas_reg['duid pre'].map(states)

#%% Merge the tables on duid pre and Timestamp
data = pd.merge(fcas_reg,dispatchload,how='inner',on=['Timestamp','duid pre'])
data = pd.merge(data,rrp,how='inner',on=['Timestamp','REGIONID'])

#%%
