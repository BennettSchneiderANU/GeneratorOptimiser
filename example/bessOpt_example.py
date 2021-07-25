# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:39:04 2021

@author: benne
"""
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
workingpath = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\test" # modify this to be your working directory
t0 = dt.datetime(2020,6,27)
t1 = dt.datetime(2020,7,1)
nem = go.NEM(workingpath)

#%% Read in the price data
data_path = os.path.join(os.path.dirname(__file__),r"RRP_example.csv") # this is the full file path
RRP = pd.read_csv(data_path,index_col=0) # read in
RRP.index = pd.to_datetime(RRP.index) # set the index to datetime
RRP.index.name = 'Timestamp' # name the index according to what our Network object expects

# load modified rrp into nem
nem.loadRaw(RRP, 5, 'Price')

#%% Instantiate BESS objects
genList = [f"bess{num}" for num in range(11)] # these are just the default set of names from the default config
# Change this to a path of your choice if you wish to customise inputs.csv for different projects
configpath = None # os.path.join(workingpath,'inputs.csv') 
region = 'NSW1' # NSW1, QLD1, SA1, TAS1, or VIC1

Generators = {}
for generator in genList:
    Generators[generator] = go.BESS(workingpath,region,generator,t0,t1,configpath=configpath)
#%% Run optimisation
m = Model(sense='MAX') # instantiate mip model object
for generator,Generator in Generators.items():
    Generator.optDispatch(nem,m)

#%% Plot results
for genToPlot in Generators.values():
    genToPlot.plotDispatch(nem,y2lim=[0,200])