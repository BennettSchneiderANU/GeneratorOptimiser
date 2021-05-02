# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:33:52 2021

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
# /mnt/wls2/fcas/old_individual_files/
#%% Extract data from one hdf5 file so we can get a better record
file_path = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\fcas_reg\FCAS_202102282355.hdf5"
f = h5py.File(file_path, 'r')
#%%
values = pd.Series([val[0] for val in f['df']['block1_values'][:]],index=f['df']['axis1'][:],name=f['df']['block1_items'][:][0])
df = pd.DataFrame(f['df']['block2_values'][:],columns=f['df']['block2_items'][:],index=values.index)
df = pd.concat([df,values],axis=1)
df.columns = [col.decode('utf-8') for col in df.columns]
df['Timestamp'] = [dt.datetime.fromtimestamp(int(epoch_time)*10**-9) for epoch_time in df.index]
df['Timestamp'] -= pd.Timedelta(11,unit='Hours') # unclear why the time correction is 11 hours, but it is, based on comparison with other raw data
df = df.set_index('Timestamp')

# print(df[df['element']==20022].loc['2021-03-01 10:30:00'])
#%%

def getNEMweb(url,data_path,num=None,filt='left'):
    """
    Goes to url, pulls a list of all the available zip files, compares with the list already downloaded to data_path
    and downloads the remainder.
    
    Args:
        url (str): url to a nemweb address.
        
        data_path (str): path where your data is downloaded using wget.
        
        num (int. Default=None): Number of new files to get out of the list of available.
        
        filt (str. Default='left'): If 'left', applied [num:], if 'right', applies [:num] filter
        
    Returns:
        toget (list of strs): List of filenames you've just downloaded (just the name and extension, not the full path)
        
    Created on 22/02/2021 by Bennett Schneider
    """
    wget.download(url, out=data_path)
    wget_file = os.path.join(data_path,'download.wget')
    with open(wget_file, "r") as file:
        text = file.readlines() # read in the contents of the page
        file.close()
    os.remove(wget_file) # remove the wget file
    # Get the list of files
    text = text[2].split('HREF=')
    myList = [txt.split('>')[0].replace('"','').split('/')[-1] for txt in text]
    myList = [ml for ml in myList if 'zip' in ml]
    
    # See what we already have in the folder
    existing = os.listdir(data_path)
    toget = [file for file in myList if file not in existing]

    # Download remaining
    print(f"Downloading to {data_path}:")
    if num:
        try:
            if filt == 'right':
                toget = toget[:num]
            elif filt == 'left':
                toget = toget[num:]
            else:
                sys.exit(f'filt must be either "left" or "right". You entered {filt}')
        except IndexError:
            pass
    if len(toget) == 0:
        print(f'Data from {url} is up to date!')
    for file in toget:
        print(file)
        wget.download(os.path.join(url,file),out=data_path)
    
    return toget

def unzipcsvs(data_path,files,**kwargs):
    """
    Unzip files containing csvs and store them in a list of pandas dataframes.
    
    Args:
        data_path (str): Location of the zip files you want to read.
        
        file (list of strs): List of filenames you want to read.
        
        **kwargs (inputs): Optional inputs for pd.read_csv()
        
    Returns:
        data (list of pandas dataframes): List of dataframes read in from csvs inside the zip files.
    
    Created on 22/03/2021 by Bennett Schneider
        
    """
    # Number of files you want to grab
    data = []
    count = 0
    for file in files:
        count += 1
        print(f"Processing {file}... ({count} of {len(files)})")
        filepath = os.path.join(data_path,file) # path to a given zip file
        
        # Create folder to unzip into
        proc_path = os.path.join(data_path,'processed')
        try:
            os.mkdir(proc_path)
        except FileExistsError:
            pass
        # Unzip
        with zipfile.ZipFile(filepath,'r') as zip_ref:
            zip_ref.extractall(proc_path) # unzip each of the zip files to the processed folder
        
        # Read in the data from the processed folder
        for data_file in os.listdir(proc_path):
            
            if '.zip' in data_file: # apply recursively
                data.extend(unzipcsvs(proc_path,[data_file],**kwargs))
            else:
                data.append(pd.read_csv(os.path.join(proc_path,data_file),**kwargs))
            
        # Delete the folder we just made and all its contents
        shutil.rmtree(proc_path)
        
    return data

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
#%% Define paths where data is stored/should be downloaded
path_fcas = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\fcas_reg"  
path_disload = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\dispatchload"
path_prices = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\prices"
#%% Download list of available files from the url
url_fcas = r"http://www.nemweb.com.au/Reports/Current/Causer_Pays/"
toget = getNEMweb(url_fcas,path_fcas)

#%%
url_disload = r"https://nemweb.com.au/Reports/Current/Next_Day_Dispatch/"
toget = getNEMweb(url_disload,path_disload,num=-40,filt='left')

#%%
url_prices = r"https://nemweb.com.au/Reports/Archive/Public_Prices/"
toget = getNEMweb(url_prices, path_prices)

#%% Choose which 4s data to read in from zips
# Define some time filter
start = dt.datetime(2021,3,1)
end = dt.datetime(2021,3,2)
days_targ = 2
date_format = "%Y%m%d%H%M"


# Construct a pandas series where the index is the timestamp and the value is the filename
files_fcas = pd.Series({
    dt.datetime.strptime(fcas_dir.split('_')[1].split('.')[0],date_format):fcas_dir for fcas_dir in os.listdir(path_fcas) if '.zip' in fcas_dir
    }).sort_index()

fcas_list = filter_series(files_fcas,start,end,30)

files_fcas = files_fcas[files_fcas.isin(fcas_list)] # filter the fcas series by the list
files_fcas.name = 'Files'
files_fcas.index.name = 'Timestamp'
files_fcas = files_fcas.to_frame().reset_index()
files_fcas['Days'] = files_fcas['Timestamp'].apply(lambda x: dt.datetime(x.year,x.month,x.day))

days = set(files_fcas['Days']) # unique set of days
days_num = len(days) # number of days in our set
days_remove = days_num - days_targ # number of days we need to remove

# Remove days randomly
for d in range(0,days_remove):
    days.remove(random.choice(tuple(days)))

files_fcas = files_fcas[files_fcas['Days'].isin(days)] # only keep files in the nominated days

print(f"\nDays to read: {len(days)}")
print(f"Files to read: {len(files_fcas)}")
#%%
fcas_5min_all = []
for day in days:
    toRead = list(files_fcas[files_fcas['Days'] == day]['Files'])
    # #%% Unzip the 4s data
    print("Unzip the 4s data")
    data_fcas = unzipcsvs(path_fcas,toRead,header=None)
    data_fcas = pd.concat(data_fcas)
    data_fcas.columns = ['Timestamp','Element number','Variable number','Value','Value quality']
    
    # #%% Enrich data with lookups
    print("Enrich data with lookups")
    variables = pd.read_csv(os.path.join(path_fcas,'820-0079 csv.csv'),header=None,index_col=0)[1]
    elements = pd.read_csv(os.path.join(path_fcas,'Elements_FCAS_202101201345.csv'),header=None,index_col=0)
    elements.iloc[:,0] = elements.iloc[:,0].apply(lambda x:x.strip(' ').split('.')[-1]) # strip whitespace and pull out unit ID
    data_fcas['Variable'] = data_fcas['Variable number'].map(variables)
    
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
    bess_fcas = data_fcas.copy()[data_fcas['Element number'].isin(list(element_map.index))]
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
    
    # #%% Get the corresponding file from dispatchload
    print("Get the corresponding file from dispatchload")
    date_format = "%Y%m%d"
    # files_disload = [disload_dir for disload_dir in os.listdir(path_disload) if '.zip' in disload_dir][:num]
    files_disload = pd.Series({
        dt.datetime.strptime(disload_dir.split('_')[-2],date_format):disload_dir for disload_dir in os.listdir(path_disload) if '.zip' in disload_dir
        }).sort_index()

    files_disload = files_disload[files_disload.index == day]
    
    toRead = list(files_disload)
    
    data_disload = unzipcsvs(path_disload,toRead,header=1)
    # #%% Process the raw data to look like the actual dispatchload table
    print("Process the raw data to look like the actual dispatchload table")
    dispatchload = []
    for disdata in data_disload:
        disdata = disdata[disdata['UNIT_SOLUTION']=='UNIT_SOLUTION'].iloc[:,4:60]
        dispatchload.append(disdata)
    
    dispatchload = pd.concat(dispatchload) # concatenate together
    dispatchload = dispatchload[dispatchload['INTERVENTION'] == 0] # for dispatch. For price, you keep the 1s
    
    # Clean up the data types
    for col in dispatchload.columns:
        if col == 'SETTLEMENTDATE':
            dispatchload[col] = pd.to_datetime(dispatchload[col])
        else:
            try:
                dispatchload[col] = pd.to_numeric(dispatchload[col])
            except ValueError:
                pass
    # #%% Apply the same duid processing to dispatchload as for fcas data
    print("Apply the same duid processing to dispatchload as for fcas data")
    dispatchload_bess = dispatchload.copy()[dispatchload['DUID'].isin(list(element_map['DUID']))]
    dispatchload_bess['duid pre'] = dispatchload_bess['DUID'].map(element_map.set_index('DUID')['pre'])
    dispatchload_bess['duid post'] = dispatchload_bess['DUID'].map(element_map.set_index('DUID')['post'])
    dispatchload_bess = dispatchload_bess.dropna(how='all',axis=1)
    dispatchload_bess = dispatchload_bess.drop(['TRADETYPE','DISPATCHINTERVAL','CONNECTIONPOINTID','DISPATCHMODE','AGCSTATUS','LASTCHANGED','DUID','RAMPDOWNRATE','RAMPUPRATE','SEMIDISPATCHCAP'],axis=1)
    # dispatchload_bess = dispatchload_bess.set_index(['SETTLEMENTDATE','RUNNO','INTERVENTION'])
    dispatchload_bess.loc[dispatchload_bess['duid post'] == 'L1',['INITIALMW','TOTALCLEARED']] *= -1
    dispatchload_bess = dispatchload_bess.groupby(['SETTLEMENTDATE','RUNNO','INTERVENTION','duid pre']).sum().reset_index()
    # #%% Map the enablement to the reg MW component
    print("Map the enablement to the reg MW component")
    fcas_5min = fcas_5min.set_index(['Timestamp','duid pre'])
    fcas_5min['prikey'] = fcas_5min.index
    fcas_5min = fcas_5min.reset_index()
    
    for col in ['RAISEREG','LOWERREG']:
        fcas_5min[col] = fcas_5min['prikey'].map(dispatchload_bess.set_index(['SETTLEMENTDATE','duid pre'])[col])
    
    # #%% Calculate the regulation dispatch fraction
    print("Calculate the regulation dispatch fraction")
    fcas_5min['RAISE_frac'] = fcas_5min[fcas_5min['GenRegComp_MW'] > 0]['GenRegComp_MW']/fcas_5min['RAISEREG']
    fcas_5min['LOWER_frac'] = fcas_5min[fcas_5min['GenRegComp_MW'] < 0]['GenRegComp_MW']/fcas_5min['LOWERREG']
    
    fcas_5min_all.append(fcas_5min)

fcas_5min_all = pd.concat(fcas_5min_all)
fcas_5min_all = fcas_5min_all.replace(np.inf,np.nan)

#%% Calculate dispatch fraction and apply to fcas_5min_final
fcas_5min_final = fcas_5min_all.copy()

# Filter out bad values
for col in ['RAISE_frac','LOWER_frac']:
    fcas_5min_final.loc[fcas_5min_final[col].abs() > 1,col] = np.nan
    

fcas_groups = []
for i,duid_group in fcas_5min_final.groupby('duid pre'):
    data = duid_group.copy()
    for col in ['RAISEREG','LOWERREG']:
        data[col] = data[col]/data[col].max() # normalise reg bids
        data = data[data[col] != 0] # exclude periods when not bid in to the reg marget
        fcas_groups.append(data)
fcas_5min_final = pd.concat(fcas_groups)


#%% Read the price data
# prices are monthly
start = fcas_5min_final['Timestamp'].min()
start = dt.datetime(start.year,start.month,1)

end = fcas_5min_final['Timestamp'].max() + rdelta.relativedelta(months=1)
end = dt.datetime(end.year,end.month,1)
#%%
date_format = "%Y%m%d"

files_prices = pd.Series({
    dt.datetime.strptime(prices_dir.split('_')[-1].split('.')[0],date_format):prices_dir for prices_dir in os.listdir(path_prices) if '.zip' in prices_dir
    }).sort_index()

files_prices = files_prices[(files_prices.index >= start) & (files_prices.index < end)]

toRead = list(files_prices)

data_prices = unzipcsvs(path_prices,toRead,header=1,parse_dates=True,error_bad_lines=False)

data_prices = pd.concat(data_prices)
#%% Clean the dataframe
prices_df = data_prices.copy()[data_prices['DREGION'] == 'DREGION'] # remove data from TREGION table

index = ['INTERVENTION','REGIONID','SETTLEMENTDATE','RUNNO']
cols = copy.deepcopy(index)
cols.extend([col for col in data_prices if 'RRP' in col])
prices_df = prices_df[cols].drop_duplicates().dropna(how='all') # remove duplicates
prices_df['SETTLEMENTDATE'] = pd.to_datetime(prices_df['SETTLEMENTDATE'])

# Stack by Market
# prices_df = prices_df.set_index(index).stack().reset_index().rename({f"level_{len(index)}":'Market',0:'Value'},axis=1)
# fig = px.line(prices_df,x='SETTLEMENTDATE',y='Value',color='Market',facet_row='REGIONID',facet_col='INTERVENTION')
# plot(fig)
#%% Map prices on to fcas_5min_final
# map regionid
regions = {'BALB': 'VIC1','GANNB': 'VIC1','HPR': 'SA1', 'LBB': 'SA1'}
fcas_5min_final['REGIONID'] = fcas_5min_final['duid pre'].map(regions)

fcas_5min_final = fcas_5min_final.set_index(['Timestamp','REGIONID'])
fcas_5min_final['prikey'] = fcas_5min_final.index
fcas_5min_final = fcas_5min_final.reset_index()
    
# map price based on timestamp and regionid
price_cols = [col for col in prices_df if 'RRP' in col]
for col in price_cols:
    fcas_5min_final[col] = fcas_5min_final['prikey'].map(prices_df.set_index(['SETTLEMENTDATE','REGIONID'])[col])

#%% Plot the 5min data
multiindex = ['Timestamp','duid pre']
toPlot = fcas_5min_final.copy().set_index(multiindex)[['RAISEREG','LOWERREG','RAISE_frac','LOWER_frac']].stack().reset_index().rename({f'level_{len(multiindex)}':'Variable',0:'Value'},axis=1)
fig = px.scatter(toPlot,x='Timestamp',y='Value',color='duid pre',facet_row='Variable').update_yaxes(matches=None)
plot(fig)

#%% Plot histogram
toPlot_hist = toPlot.copy() #toPlot[toPlot['Variable'].isin(['RAISE_frac','LOWER_frac'])]
fig = px.histogram(toPlot_hist,x='Value',color='duid pre',facet_row='Variable').update_yaxes(matches=None).update_layout(xaxis={'range':[-1,1]})
plot(fig)

#%% Plot fraction against price
fig = px.scatter(fcas_5min_final,x='LOWERREGRRP',y='LOWER_frac',color='duid pre')
plot(fig)

#%% Plot diurnal bar chart
toPlot_diurnal = toPlot.copy()
toPlot_diurnal['Hour'] = pd.DatetimeIndex(toPlot_diurnal['Timestamp']).hour
fig = px.box(toPlot_diurnal,x='Hour',y='Value',color='duid pre',facet_row='Variable').update_yaxes(matches=None)
plot(fig)

#%% Plot slices of the raw 4s data
plot_data = bess_fcas.copy()
plot_data['Timestamp'] = pd.to_datetime(plot_data['Timestamp'])

# start = dt.datetime(2021,1,19,6,30) # choose a date slice
# end = dt.datetime(2021,1,19,7,30)
# toPlot = plot_data[(plot_data['Timestamp'] > start) & (plot_data['Timestamp'] <= end)]

#%%
fig = px.line(toPlot,x='Timestamp',y='Value',color='duid pre',facet_row='Variable',height=1200).update_yaxes(matches=None)
plot(fig)