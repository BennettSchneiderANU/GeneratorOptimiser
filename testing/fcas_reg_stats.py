# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:10:01 2021

@author: benne
"""
#%%
import sys
import os

myPath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'src')
sys.path.append(myPath)

import genOpt as go
import datetime as dt
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


#%% Load dependencies
rrp_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\prices\20210505_NEM_20200701-00-05_20210301-00-00_RRP.csv"
dispatchload_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\dispatchload\20210507_NEM_20200701-00-05_20210301-00-00_DISPATCHLOAD.csv"
fcas_reg_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\fcas_reg\reg_component_5min.csv"
export = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\results"

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

#%% Clean up fcas_reg and map on states
fcas_reg = fcas_reg_raw.copy()

fcas_reg = fcas_reg.groupby(['Timestamp','duid pre']).sum().reset_index() # there are duplicate entries for the same duid and ts

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

#%% Calculate the regulation dispatch fraction
data['RAISE_frac'] = (data[data['GenRegComp_MW'] > 0]['GenRegComp_MW']/data['RAISEREG']).replace(np.inf,np.nan)
data['LOWER_frac'] = (data[data['GenRegComp_MW'] < 0]['GenRegComp_MW']/data['LOWERREG']).replace(np.inf,np.nan)

# Clean up bad values
data.loc[data['RAISE_frac'].abs() > 10,'RAISE_frac'] = np.nan
data.loc[data['LOWER_frac'].abs() > 10,'LOWER_frac'] = np.nan

#%% Add intervals
bins = {
        'RRP':(-1000,-300,-100,0,100,300,1000,5000,15000),
        'RAISEREGRRP':(0,100,300,1000,5000,15000),
        'LOWERREGRRP':(0,100,300,1000,5000,15000)
        }

for col in [col for col in data.columns if 'RRP' in col]:
    try:
        data[f"{col}_interval"] = pd.Series(list(pd.cut(data[col],bins=bins[col]))).astype(str)
    except KeyError:
        pass

data['Hour'] = pd.DatetimeIndex(data['Timestamp']).hour

# Flag data before/after Oct 1 2020 (introduction of MPF)
data['MPF'] = 'No'
data.loc[data['Timestamp'] > dt.datetime(2020,10,1),'MPF'] = 'Yes'

#%% Process data to plot
multiindex = ['Timestamp','duid pre','Hour']
toPlot = data.copy().set_index(multiindex)[['RAISEREG','LOWERREG','RAISE_frac','LOWER_frac','GenRegComp_MW']].stack().reset_index().rename({f'level_{len(multiindex)}':'Variable',0:'Value'},axis=1)
# Flag data before/after Oct 1 2020 (introduction of MPF)
toPlot['MPF'] = 'No'
toPlot.loc[toPlot['Timestamp'] > dt.datetime(2020,10,1),'MPF'] = 'Yes'
#%% Plot the 5min data
fig = px.scatter(toPlot,x='Timestamp',y='Value',color='duid pre',facet_row='Variable').update_yaxes(matches=None)
plot(fig)

#%% Plot histogram
toPlot_hist = toPlot[toPlot['Variable'].isin(['RAISE_frac','LOWER_frac'])]

fig = px.histogram(toPlot_hist,x='Value',color='duid pre',facet_row='Variable',facet_col='Hour',width=3000).update_yaxes(matches=None).update_layout(xaxis={'range':[-1,1]})
plot(fig)

#%% Interval box plot
multiindex = ['Timestamp','duid pre','RRP_interval']
toPlot_box = data.copy().set_index(multiindex)[['RAISE_frac','LOWER_frac']].stack().reset_index().rename({f'level_{len(multiindex)}':'Variable',0:'Value'},axis=1)
fig = px.box(toPlot_box,x='RRP_interval',y='Value',color='duid pre',facet_row='Variable') # no price/Fr relationship
plot(fig)
#%% Plot fraction against price
fig = px.scatter(data,x='RRP',y='LOWER_frac',color='duid pre')
plot(fig)

#%% Plot diurnal bar chart
toPlot_diurnal = toPlot.copy()
toPlot_diurnal['Hour'] = pd.DatetimeIndex(toPlot_diurnal['Timestamp']).hour
fig = px.box(toPlot_diurnal,x='Hour',y='Value',color='duid pre',facet_row='Variable').update_yaxes(matches=None)
plot(fig)

#%% Enablement-weighted diurnal average Fr 
multiindex = ['duid pre','Hour','MPF']
data['wRAISE_frac'] = data['RAISEREG']*data['RAISE_frac']
Fr_raise = (data.groupby(multiindex)['wRAISE_frac'].sum()/data.groupby(multiindex)['RAISEREG'].sum()).reset_index().rename({0:'RAISE'},axis=1)

data['wLOWER_frac'] = data['LOWERREG']*data['LOWER_frac']
Fr_lower = (data.groupby(multiindex)['wLOWER_frac'].sum()/data.groupby(['duid pre','Hour'])['LOWERREG'].sum()).reset_index().rename({0:'LOWER'},axis=1)

Fr = pd.merge(Fr_raise,Fr_lower,how='inner',on=multiindex).set_index(multiindex).stack().reset_index().rename({f'level_{len(multiindex)}':'Market',0:'Value'},axis=1)

fig = px.line(Fr,x='Hour',y='Value',color='duid pre',facet_col='Market',facet_row='MPF').update_yaxes(matches=None)
plot(fig)
fig.write_html(os.path.join(export,'hourly_Fr.html'))