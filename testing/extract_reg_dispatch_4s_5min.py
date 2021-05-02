# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:30:35 2021

@author: benne
"""
import h5py
import datetime as dt
import sys
import os
import pandas as pd
import numpy as np
#%%

def aggEnd(data,inFreq,outFreq,func=np.sum,kind='timestamp'):
    """
    Wrapper around pandas.DataFrame.agg() that specifically applied period-ending convention.

    Args:
        data (pandas DataFrame): Any dataframe with a datetimeindex as the index.

        inFreq (str): Any standard time format as suitable for resample that denotes the frequency of the input data.

        outFreq (str): Any standard time format as suitable for resample that denotes the frequency of the output data.

        func (function, str, list or dict): Function to use for aggregating the data. If a function, must either work when passed a DataFrame or when passed to DataFrame.apply.
            Accepted combinations are:
              - function
              - string function name
              - list of functions and/or function names, e.g. [np.sum, 'mean']
              - dict of axis labels -> functions, function names or list of such.
              
        kind (str. Default=None): pandas.DataFrame.resample() input. Either 'period' or 'timestamp'. Pass ‘timestamp’ to convert the resulting 
            index to a DateTimeIndex or ‘period’ to convert it to a PeriodIndex. By default we use 'timestamp'.
            Use 'period' for monthly/yearly resamples!
            
    Returns:
        data (pandas DatFrame): data, aggregated according to outFreq. Can be aggregated using different functions according to pandas.DataFrame.agg().

    Use this function to flexibly aggregate a dataframe according to period-ending convention.
    
    Created on 28/01/2021 by Bennett Schneider

    """
    data = data.copy() # copy input data to avoid messing with the input
    data = data.shift(-1,freq=inFreq) # shift back by 1
    data = data.resample(outFreq,kind=kind).agg(func) # resample and aggregate
    if kind == 'timestamp':
        data = data.shift(1,freq=outFreq) # shift forward by 1

    return data


def filter_series(series,start,end,freq):
    """
    Returns a list of elements from a series where the index lies between start and end.
    
    Args:
        series (pandas Series): Index must be datetimeindex
        
        start (datetime): Start of closed interval.
        
        end (datetime): End of closed interval.
        
        freq (int): Frequency of series.index in minutes.
    
    Returns:
        mylist (list): List of elements of series with index between start and end.
        
    Created on 24/04/2021 by Bennett Schneider
        
    """
    mylist = list(series[(series.index >= start) & (series.index <= end)])
    expected_intervals = ((end - start).total_seconds())/(freq*60)
    print(f"You have {100*len(mylist)/expected_intervals:.2f}% of the data across the following intervals:\n{len(mylist)}\n{end - start}")
    
    return mylist
    
def match_set(df,col,match):
    myset = set(df[col])
    if myset == match:
        return df  
    
#%% Read the hdf5 files and convert to 5mins
# folder_actual = r"/mnt/wls2/fcas/old_individual_files/"
# files_actual = os.listdir(folder_actual)

folder =  r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\fcas_reg"
files = [r"FCAS_202102282355.hdf5"]
for file in files:
    file_path = os.path.join(folder,file)
    print(f"Extract data from compressed file: {file}")
    f = h5py.File(file_path, 'r')
    values = pd.Series([val[0] for val in f['df']['block1_values'][:]],index=f['df']['axis1'][:],name=f['df']['block1_items'][:][0])
    df = pd.DataFrame(f['df']['block2_values'][:],columns=f['df']['block2_items'][:],index=values.index)
    df = pd.concat([df,values],axis=1)
    df.columns = [col.decode('utf-8') for col in df.columns]
    df['Timestamp'] = [dt.datetime.fromtimestamp(int(epoch_time)*10**-9) for epoch_time in df.index]
    df['Timestamp'] -= pd.Timedelta(11,unit='Hours') # unclear why the time correction is 11 hours, but it is, based on comparison with other raw data
    
    df = df.rename({'element':'Element number','variable':'Variable number','value':'Value','quality':'Value quality'},axis=1)
    
    print("Enrich data with lookups")
    variables = pd.read_csv(os.path.join(folder,'820-0079 csv.csv'),header=None,index_col=0)[1]
    elements = pd.read_csv(os.path.join(folder,'Elements_FCAS_202101201345.csv'),header=None,index_col=0)
    elements.iloc[:,0] = elements.iloc[:,0].apply(lambda x:x.strip(' ').split('.')[-1]) # strip whitespace and pull out unit ID
    df['Variable'] = df['Variable number'].map(variables)
    
    # #%% Filter for bess duids and map on duid names 
    print("Filter for bess duids and map on duid names")
    # Negative GenRegComp_MW gets assigned to the generator for batteries, even when due to a load participating in lower.
    # Need to combine gens and loads into a single duid where negative is load, positive is gen  
    # Remove elements that don't have G1 L1 pairs as their suffix
    element_map = pd.DataFrame()
    element_map['pre'] = elements.iloc[:,0].apply(lambda x: x[:-2])
    element_map['post'] = elements.iloc[:,0].apply(lambda x: x[-2:])
    
    element_map = element_map.groupby('pre').apply(lambda pair: match_set(pair,'post',{'L1','G1'})).dropna()
    element_map.index.name = 'Element number'
    element_map['DUID'] = element_map.sum(axis=1)
    
    # #%% Map the duids and filter out non-battery elements
    print("Map the duids and filter out non-battery elements")
    # Remove data not in the element map and then map on the partial duids
    bess_fcas = df.copy()[df['Element number'].isin(list(element_map.index))]
    bess_fcas['duid pre'] = bess_fcas['Element number'].map(element_map['pre']) # map prefix of duid which conflates the load and generator portion of a bess
    bess_fcas['duid post'] = bess_fcas['Element number'].map(element_map['post'])
    
    # Set sign based on duid post, then group by duid pre
    bess_fcas.loc[bess_fcas['duid post'] == 'L1','Value'] *= -1 # set loads to negative
    bess_fcas = bess_fcas[['Timestamp','duid pre','Variable','Value']].groupby(['Timestamp','duid pre','Variable']).sum().reset_index()
    # #%% Create a new variable called the 'non_reg_component' and append it to data_fcas
    print("Create a new variable called the 'non_reg_component' and append it to data_fcas")
    reg_component = bess_fcas[bess_fcas['Variable'] == 'GenRegComp_MW'].set_index(['Timestamp','duid pre'])
    gen_component = bess_fcas[bess_fcas['Variable'] == 'Gen_MW'].set_index(['Timestamp','duid pre'])
    
    non_reg_component = reg_component.copy()
    non_reg_component['Variable'] = 'GenNonRegComp_MW'
    non_reg_component['Value'] = gen_component['Value'] - reg_component['Value']
    
    bess_fcas = bess_fcas.append(non_reg_component.reset_index())
    
    # #%% Aggregate the GenRegComp_MW to 5min, grouped by duid
    print("Aggregate the GenRegComp_MW to 5min, grouped by duid")
    variables = ['GenRegComp_MW']
    fcas_slice = bess_fcas[bess_fcas['Variable'].isin(variables)]
    fcas_pivot = pd.pivot_table(fcas_slice,index=['Timestamp','duid pre'],columns='Variable').reset_index()
    # Map the duid and the duid prefix onto fcas_pivot
    fcas_pivot['Timestamp'] = pd.to_datetime(fcas_pivot['Timestamp'])
    fcas_pivot.columns = [(col1 + col2).replace('Value','') for col1,col2 in zip(fcas_pivot.columns.get_level_values(0),fcas_pivot.columns.get_level_values(1))]
    
    # #%% Average to 5mins
    print("Average to 5mins")
    fcas_5min = fcas_pivot.set_index('Timestamp').groupby('duid pre').apply(lambda dslice: aggEnd(dslice,'4s','5T',func=np.mean)).reset_index()
    