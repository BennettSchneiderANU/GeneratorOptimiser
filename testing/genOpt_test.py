import genOpt as go
import datetime as dt
import sys
import os
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from mip import *
#%% Initialise
t0 = dt.datetime(2020,1,1)
t1 = dt.datetime(2020,2,1)
region = 'NSW1'
path = os.path.dirname(os.path.realpath(__file__))
nem = go.NEM(path)
m = Model(sense='MAX')
#%%
data_path = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\RRP.csv"
RRP = pd.read_csv(data_path,index_col=0)
RRP.index = pd.to_datetime(RRP.index) # [dt.datetime.strptime(DT,'%Y-%m-%d %H:%M:%S') for DT in RRP.index]
RRP.index.name = 'Timestamp'
#%% Create price input and run optimiser
# load modified rrp into nem
nem.loadRaw(RRP, 5, 'Price')
# nem.procPrice(30, region, t0, t1,modFunc=bess.modFunc,**bess.kwargs)
#%%
bess1 = go.BESS(path,region,scenario='RRP',modFunc=go.thresh_smooth,window=4,thresh=40,roll=True)
bess2 = go.BESS(path,region,scenario='RRP',modFunc=None)
#%%
# run optimised dispatch usinf preload
bess1.optDispatch(nem, m, t0, t1)
#%%
# RRP_orig = RRP.copy()
# RRP_orig['modFunc'] = 'orig'
# bess.pickle()
#%%
toPlot = rrp_mod.copy()
toPlot.index.name = 'Timestamp'
toPlot.reset_index(inplace=True)
#%%

fig = px.line(toPlot,x='Timestamp',y='RAISEREGRRP',color='modFunc')
plot(fig,auto_open=True)