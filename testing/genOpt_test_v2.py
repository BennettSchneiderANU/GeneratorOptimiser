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
from plotly.subplots import make_subplots
from mip import *
import time
from ipywidgets import widgets
go.SetLogging(path = '', level = 'DEBUG')

#%% Initialise
path_base = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working"
t0 = dt.datetime(2020,6,27)
t1 = dt.datetime(2020,7,1)
region = 'NSW1'
myPath = os.path.join(path_base,r"results\bessOpt")
nem = go.NEM(myPath)

#%% Read in the price data
data_path = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\RRP_test.csv"
RRP = pd.read_csv(data_path,index_col=0)
RRP.index = pd.to_datetime(RRP.index) # [dt.datetime.strptime(DT,'%Y-%m-%d %H:%M:%S') for DT in RRP.index]
RRP.index.name = 'Timestamp'
# load modified rrp into nem
nem.loadRaw(RRP, 5, 'Price')


#%% Instantiate BESS objects

genList = [f"bess{num}" for num in range(11)]
Generators = {}
for generator in genList:
    Generators[generator] = go.BESS(path,region,generator)
#%% Run optimisation
m = Model(sense='MAX')


for generator,Generator in Generators.items():
    Generator.optDispatch(nem,m,t0,t1)

#%%
for genToPlot in Generators.values():
# genToPlot = Generators['bess3']
    genToPlot.plotDispatch(nem,t0=dt.datetime(2020,6,27),t1=dt.datetime(2020,7,1),y2lim=[0,200])

#%%
# Optimiser looks basically fine. Functionality to run optimisation on the same bess object across different time periods
# across different runs is still super buggy. Something wrong with getRRP(). Might not be a good idea to allow this anyways.
# Could just remove the functionality by setting t0 and t1 as attributes, which would make things simpler



