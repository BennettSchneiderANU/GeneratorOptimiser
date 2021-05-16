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
rrp_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\prices\20210514_NEM_20190501-00-05_20210501-00-00_RRP.csv"
dispatchload_file = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\dispatchload\20210514_NEM_20190501-00-05_20210501-00-00_DISPATCHLOAD.csv"
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
data.loc[data['RAISE_frac'].abs() > 1,'RAISE_frac'] = np.nan
data.loc[data['LOWER_frac'].abs() > 1,'LOWER_frac'] = np.nan

#%% Add intervals

# Price intervals
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


# Fr intervals
for market in ['RAISE','LOWER']:
    # Set market_frac to 0 if nan during a period where the duid is enabled in the corresponding market
    data.loc[data[f'{market}REG']>0,f'{market}_frac'] = data[f'{market}_frac'].fillna(0)
    
    data.loc[abs(data[f'{market}_frac']) < 0.01,f'{market}_frac'] = 0 # set really small values to 0 to clean up bins
    # data[f'{market}_frac_bins'] = pd.Series(list(pd.cut(data[f'{market}_frac'],bins=10,right=False))).astype(str)
    data[f'{market}_frac_bins'] = pd.IntervalIndex(pd.cut(data[f'{market}_frac'],bins=10,right=False)).mid

# Calculate revenue due to regulation energy dispatch in RAISE and LOWER respectively
data['RAISE_Energy_Revenue'] = data['RAISE_frac']*data['RAISEREG']*data['RRP']
data['LOWER_Energy_Revenue'] = data['LOWER_frac']*data['LOWERREG']*data['RRP']

# Calculate the maximum total revenue due to regulation dispatch in RAISE and LOWER respectively
data['RAISE_Energy_Revenue_max'] = data['RAISEREG']*data['RRP']
data['LOWER_Energy_Revenue_max'] = -data['LOWERREG']*data['RRP']


data['Hour'] = 3*(pd.DatetimeIndex(data['Timestamp']).hour//3)
data['Month'] = pd.DatetimeIndex(data['Timestamp']).month

# Flag data before/after Oct 1 2020 (introduction of MPF)
data['MPF'] = 'No'
data.loc[data['Timestamp'] > dt.datetime(2020,10,1),'MPF'] = 'Yes'
data_crop = data.copy()[data['Month'].isin([10,11,12,1,2])]

#%% Process data to plot
multiindex = ['Timestamp','duid pre','Hour','Month']
toPlot = data_crop.copy().set_index(multiindex)[['RAISEREG','LOWERREG','RAISE_frac','LOWER_frac','GenRegComp_MW']].stack().reset_index().rename({f'level_{len(multiindex)}':'Variable',0:'Value'},axis=1)
# Flag data before/after Oct 1 2020 (introduction of MPF)
toPlot['MPF'] = 'No'
toPlot.loc[toPlot['Timestamp'] > dt.datetime(2020,10,1),'MPF'] = 'Yes'
#%% Plot the 5min data
fig = px.scatter(toPlot,x='Timestamp',y='Value',color='duid pre',facet_row='Variable').update_yaxes(matches=None)
plot(fig)

#%% Plot histogram
toPlot_hist = toPlot[toPlot['Variable'].isin(['RAISE_frac','LOWER_frac'])]

fig = px.histogram(toPlot_hist,x='Value',color='duid pre',facet_row='Variable',facet_col='MPF').update_yaxes(matches=None).update_layout(xaxis={'range':[-1,1]})
fig.write_html(os.path.join(export,f"mpf_Fr_hist.html"))
plot(fig)

#%% Interval box plot
multiindex = ['Timestamp','duid pre','RRP_interval']
toPlot_box = data.copy().set_index(multiindex)[['RAISE_frac','LOWER_frac']].stack().reset_index().rename({f'level_{len(multiindex)}':'Variable',0:'Value'},axis=1)
fig = px.box(toPlot_box,x='RRP_interval',y='Value',color='duid pre',facet_row='Variable') # no price/Fr relationship
plot(fig)
#%% Plot values against one another
fig = px.scatter(data,x='RRP',y='RAISEREG',color='duid pre')
plot(fig)

#%% Plot diurnal bar chart
# toPlot_diurnal = toPlot.copy()[toPlot['Variable'].isin(['LOWER_frac','RAISE_frac'])]
# facet_col = 'Month'
# # toPlot_diurnal['Hour'] = pd.DatetimeIndex(toPlot_diurnal['Timestamp']).hour
# fig = px.histogram(
#     toPlot_diurnal,
#     x='Value',
#     color='MPF',
#     facet_row='Variable',
#     facet_col=facet_col,
#     barmode='group',
#     nbins=20,
#     histnorm  = 'probability density'
#     )# .update_yaxes(matches=None)
# fig.write_html(os.path.join(export,f"mpf_Fr_hist_{facet_col}.html"))
# plot(fig)

# #%% Enablement-weighted diurnal average Fr 
# multiindex = ['duid pre','Hour','MPF']
# data['wRAISE_frac'] = data['RAISEREG']*data['RAISE_frac']
# Fr_raise = (data.groupby(multiindex)['wRAISE_frac'].sum()/data.groupby(multiindex)['RAISEREG'].sum()).reset_index().rename({0:'RAISE'},axis=1)

# data['wLOWER_frac'] = data['LOWERREG']*data['LOWER_frac']
# Fr_lower = (data.groupby(multiindex)['wLOWER_frac'].sum()/data.groupby(['duid pre','Hour'])['LOWERREG'].sum()).reset_index().rename({0:'LOWER'},axis=1)

# Fr = pd.merge(Fr_raise,Fr_lower,how='inner',on=multiindex).set_index(multiindex).stack().reset_index().rename({f'level_{len(multiindex)}':'Market',0:'Value'},axis=1)

# fig = px.line(Fr,x='Hour',y='Value',color='duid pre',facet_col='Market',facet_row='MPF').update_yaxes(matches=None)
# plot(fig)
# fig.write_html(os.path.join(export,'hourly_Fr.html'))

#%% Calculate revenue-weighted average Fr, as per the notes
# Cut this across any pre-defined set of groupings

group = ['duid pre'] # variable apart from Fr_bins you want to group by

def calc_bn(data_group,market,group):
    group_dist = copy.deepcopy(group)
    group_dist.append(f'{market}_frac_bins') # group by the original group, plus the Fr bins
    # apply new grouping only to the numerator
    bn = data_group.groupby(group_dist)[f'{market}_Energy_Revenue_max'].sum()/data_group[f'{market}_Energy_Revenue_max'].sum()
    # set the index so it gels with the caller
    bn = bn.reset_index().drop(group,axis=1).set_index(f'{market}_frac_bins')
    bn.columns = ['bn']
    bn['Fr'] = (bn['bn']*bn.index).sum() # this is the revenue-weighted average Fr for this group
    return bn

bn = pd.DataFrame()
for market in ['RAISE','LOWER']: # for each market
    # Calculate F_T distribution across a number of different groupings
    bn_part = data.groupby(group).apply(lambda x: calc_bn(x,market,group)).reset_index().rename({f'{market}_frac_bins':'Fr_bins'},axis=1)
    bn_part['Market'] = market
    if market == 'LOWER':
        bn_part['Fr'] *= -1
    bn = bn.append(bn_part)

# Plot revenue-weighted average
facet_row='Market'
facet_col = 'duid pre'
color=None
fig = px.bar(bn,x='Fr_bins',y=['bn','Fr'],facet_row=facet_row,facet_col=facet_col,color=color,barmode='group').update_xaxes(matches=None)
plot(fig)
#%% Save
suffix = '-'.join(e for e in [facet_row,facet_col,color] if e)
fig.write_html(os.path.join(export,f'Fr_distribution_{suffix}.html'))

