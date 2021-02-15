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
#%%
bess1 = go.BESS(path,region,scenario='Energy',modFunc=go.thresh_smooth,window=4,thresh=40,roll=True)
bess2 = go.BESS(path,region,scenario='Energy',modFunc=None)
#%%
# nem.procPrice(30, region, t0, t1,pivot=True,modFunc=bess1.modFunc,**bess1.kwargs)
# rrp,rrp_mod = bess1.getRRP(nem,t0,t1)
#%%
# run optimised dispatch usinf preload
bess1.optDispatch(nem, m, t0, t1)
bess2.optDispatch(nem, m, t0, t1)
#%%
revenue = bess1.revenue.copy().reset_index().set_index(['Timestamp','Market']).stack().reset_index()
#%%
revenue = bess1.stackRevenue()
figure = px.line(revenue,x='Timestamp',y='Value',color='Market',line_dash='Result')
# plot(figure)
sy = [d.name for d in figure.data if '$' in d.name]
go.plotlyPivot(None,figure,secondary_y=sy,y2lim=[-300,300])
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