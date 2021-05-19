import sys
import os
path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'src')
if path not in sys.path:
    sys.path.append(path)

import genOpt as go
import datetime as dt
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from mip import *
import time
#%% Initialise
path_base = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working"
t0 = dt.datetime(2020,1,1)
t1 = dt.datetime(2020,2,1)
region = 'NSW1'
myPath = os.path.join(path_base,r"results\bessOpt")
nem = go.NEM(myPath)

#%%
data_path = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\RRP.csv"
RRP = pd.read_csv(data_path,index_col=0)
RRP.index = pd.to_datetime(RRP.index) # [dt.datetime.strptime(DT,'%Y-%m-%d %H:%M:%S') for DT in RRP.index]
RRP.index.name = 'Timestamp'
#%% Create price input and run optimiser
# load modified rrp into nem
nem.loadRaw(RRP, 5, 'Price')
#%%
bess1 = go.BESS(path,region,scenario='Energy',modFunc=None)
bess2 = go.BESS(path,region,scenario='Energy',modFunc=None)
#%% Clear bess objects so they can be remade
bess1 = None
bess2 = None
# nem.procPrice(30, region, t0, t1,pivot=True,modFunc=bess1.modFunc,**bess1.kwargs)
# rrp,rrp_mod = bess1.getRRP(nem,t0,t1)
#%%
m = Model(sense='MAX')
# run optimised dispatch usinf preload
bess1.optDispatch(nem, m, t0, t1,debug=True,optfunc=go.BESS_COINOR)
bess2.optDispatch(nem, m, t0, t1,debug=True,optfunc=go.BESS_COINOR_hurdle,hurdle=20)
revenue1 = bess1.stackRevenue()
revenue2 = bess2.stackRevenue()
#%%
figure1 = px.line(revenue1,x='Timestamp',y='Value',color='Market',facet_row='Result').update_yaxes(matches=None)
plot(figure1)
time.sleep(1)
go.plotlyPivot(bess1.operations.sort_index())
#%%
figure2 = px.line(revenue2,x='Timestamp',y='Value',color='Market',facet_row='Result').update_yaxes(matches=None)
plot(figure2)
time.sleep(1)
go.plotlyPivot(bess2.operations.sort_index())

#%%
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