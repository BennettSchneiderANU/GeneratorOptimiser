#%%
###################################################################################################################
######################################### SETUP ###################################################################
###################################################################################################################

""" Enter the version number of the Windlab Python Toolbox you wish to implement"""
version = 'Dev' # version number as an int or float, or 'Dev'

"""This is just some necessary preamble to load in our toolbox"""
import sys # imported so we can set the location of the toolbox in the next line
# insert the version number into the path
if type(version) != str:
    v_str = 'Versions\\V{}'.format(version)
else:
    v_str = version

# Import the toolbox
path = "T:\\Tools\\Python\\ToolBox\\{}".format(v_str)
# path = rf"Z:\Technical\Canberra\Tools\Python\ToolBox\{v_str}"

if path not in sys.path:
    sys.path.append(path) #Add the location of the toolbox to the path so it can be imported later
import WindlabPythonToolbox as WPT # import toolbox so all the methods that are written are availeble to us
print("Finished importing WPT")

#%%
# Import other custom packages
path = r"C:\Users\bennett.schneider\OneDrive - Australian National University\Master of Energy Change\SCNC8021\Analysis\Packages"
if path not in sys.path:
    sys.path.append(path)
from analysis_functions import * 


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,optimize
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.express as px
import plotly.colors
import sqlalchemy
import pymysql
import os
import time
import random
import copy
import math


#%%
###################################################################################################################
######################################### INPUTS ##################################################################
###################################################################################################################
# WindFarm settings
# regions = ['VIC1','SA1','TAS1','NSW1','QLD1']

month = ''
quarter = ''
year = ''
t0 = '' # '14/1/2020 0:05'
t1 = '' # '20/01/2020 0:00'

interval = 'fy'
myPath = fr"C:\Users\bennett.schneider\OneDrive - Australian National University\Master of Energy Change\SCNC8021\Analysis\final"
resultsDir = "Results"
pickle_paths = fr"C:\Users\bennett.schneider\OneDrive - Australian National University\Master of Energy Change\SCNC8021\Analysis\final\NEM_RRP_FCAS_picklePaths_{interval}.csv"
forceReload = False

pickles = pd.read_csv(pickle_paths,index_col=0)

freq = 5
export = False
table = 'Daily_03'
fields = [
    'RRP',
    'RAISE6SECRRP',
    'RAISE60SECRRP',
    'RAISE5MINRRP',
    'RAISEREGRRP',
    'LOWER6SECRRP',
    'LOWER60SECRRP',
    'LOWER5MINRRP',
    'LOWERREGRRP'
    ]


NEMs = {}
# for fYear in pickles.index
for intvl in pickles.index:
    # pickle_path = pickles.loc[fYear,'path']
    pickle_path = pickles.loc[intvl,'path']

    try:
        if forceReload:
            raise FileNotFoundError
        NEM = WPT.unPickleNEM(pickle_path)
        NEM.path = myPath
    except (TypeError,FileNotFoundError) as e:
            
        if interval == 'quarter':
            quarter = intvl
        elif interval == 'month':
            month = intvl
        elif interval == 'fy':
            year = f"{intvl},{intvl+1}"
        
        NEM = WPT.NEM(myPath,month=month,quarter=quarter,year=year,t0=t0,t1=t1)
            
        # Load the RRP for each region 
        NEM.loadDaily(table=table,fields=fields,freq=freq,load=True,export=export)
     
    # pickles.loc[str(fYear),'path'] = NEM.pickleNEM()
    pickles.loc[intvl,'path'] = NEM.pickleNEM()
        
    # NEMs[str(fYear)] = NEM
    NEMs[intvl] = NEM
    
# pickles.to_csv(pickle_paths,index='FY')
pickles.to_csv(pickle_paths,index=interval)

#%%
# Check progress
for year,NEM in NEMs.items():
    print("\n********************")
    print(year)
    for region,Region in NEM.Regions.items():
        print(region)
        
        try:
            print(Region.RESULTS.keys())
            # keys = [key for key in Region.RESULTS.keys() if 'v1' in key]
            # for key in keys:
            #     Region.RESULTS.pop(key)
            # # try:
            #     myKey = 'RRP_5_all/4_all/4_0.85_0_'
            #     Region.RESULTS['RRP_5_all/4_all/4_0.85_0_v0'] = Region.RESULTS.pop(myKey)
            # except KeyError:
            #     print(f"{region}, {year} doesn't contain {myKey}")
        except AttributeError:
            print(f"No results yet")
#%%
# All regions from 2012 to 2016. Just NSW up to 2017. 2,4,8,8,10h BESS
regDist_path = os.path.join(myPath,r"regDisDist.csv")

scenarios = {
    'RRP': ['RRP'],
    'FCASCoopt': ['RRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP','RAISEREGRRP','LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP','LOWERREGRRP'],
    'RegCoopt': ['RRP','RAISEREGRRP','LOWERREGRRP'],
    'RegRaiseCoopt': ['RRP','RAISEREGRRP'],
    'RegLowerCoopt': ['RRP','LOWERREGRRP'],
    'ContCoopt': ['RRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP','LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP'],
    'ContRaiseCoopt': ['RRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP'],
    'ContLowerCoopt': ['RRP','LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP']
    }
# Settings

years = [2019] # [int(yr) for yr in np.linspace(2006,2019,14)] # [2006,2012,2019] #  [2006,2012,2019] # [int(yr) for yr in np.linspace(2006,2019,14)] # [2019] # list(NEMs.keys()) # 
eta = 1 # round-trip efficiency
regDict_versions = ['v0'] #  ['v0'] # ['fr00','fr01','fr02','fr03','fr04','fr05']
regDisFracs = [0.2]
sMaxs = [8] # 4] # [0.5,1,2,4] 
rMax = 1 # MW
runOpt = True
hr_frac = freq/60
freq = 5
# markets =  # ['RRP','RAISEREGRRP','LOWERREGRRP'] # ['RRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP','RAISEREGRRP','LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP','LOWERREGRRP']
shorthands = ['RRP'] #,'RegRaiseCoopt','RegLowerCoopt'] # ['RRP'] #  #'ContCoopt' 'RegRaiseCoopt','RegLowerCoopt','ContRaiseCoopt','ContLowerCoopt']

regions = ['NSW1'] # ,'VIC1','TAS1','QLD1'] # ['SA1','VIC1','TAS1','QLD1','NSW1'] # ['NSW1','SA1','VIC1','TAS1','QLD1']
tFcsts = {f"{h}h":h for h in [0.5,1,2,4,8,16]} # {'div':4}
tInts = {'30m':30}# {'div':4}
# div=4

smooth = False
thresh=40
hrs = 4
roll=False

if smooth:
    shorthands = [f"{sh}-{thresh}-{hrs}-{roll}" for sh in shorthands]

    
    
# Initialise the maximisation model

if runOpt:
    print('Initialising model')
    m = Model(sense='MAX')

    #for year,NEM in NEMs.items():
    for regDisFrac in regDisFracs:
        for regDict_version in regDict_versions:
            regDict = pd.read_csv(regDist_path,index_col=0).to_dict()[regDict_version] # Fraction of regulation FCAS that is dispatched on average
            # breakpoint()
            for year in years:
                NEM = NEMs[year]
                for region in regions:
                    Region = NEM.Regions[region]
                    print(region)
                    print(year)
                    for shorthand in shorthands:
                        print(shorthand)
                        rrp = Region.Daily[table].copy()
                        rrp30 = WPT.timeAvgEnd(rrp,freq,30) # average to 30mins
                        
                        for categ_FH,val_FH in tFcsts.items():
                            for categ_DI,val_DI in tInts.items():
                                if categ_DI == 'div':
                                    tFcst,tFcst_str = create_tFcst(len(rrp),hr_frac,div=val_FH)  # int(len(rrp)*hr_frac/4) # /91 # 24*365 # 20*sMax # length of the price forecast in hours
                                else:
                                    tFcst,tFcst_str = val_FH,categ_FH
                                if categ_DI == 'div':
                                    tInt,tInt_str = create_tInt(len(rrp),hr_frac,div=val_DI) # int(len(rrp)*hr_frac*60/4) # sMax*60/2  #/91  # 60*24*365 # 60*10*sMax # length of the dispatch interval in minutes
                                else:
                                    tInt,tInt_str = val_DI,categ_DI
                                # RRP = WPT.timeAvgEnd(Region.Daily[table],5,30)
                                # breakpoint()
                                # find key
                                fkey = [key for key in scenarios.keys() if key in shorthand][0]
                                # breakpoint()
                                markets = scenarios[fkey]
                                
                                if smooth:
                                    for col in rrp30.columns:
                                        if col.split('_')[0] not in markets: # only run for the markets that we care about
                                            print(f"Skipping smoothing for {col}")
                                            rrp30[f"{col}_obj"] = rrp30[col]
                                        else:
                                            print(f"Smoothing {col}")
                                            rrp30[f"{col}_obj"] = thresh_smooth(rrp30[[col]],30,hrs,thresh,col,roll=roll)
                                        # rrp.loc[rrp[col] > 400,f"{col}_smooth"] = 400
                                        # rrp.loc[rrp[col] < -400,f"{col}_smooth"] = -400
                                        # rrp.loc[(rrp[col] <= 400) & (rrp[col] >= -400),f"{col}_smooth"] = rrp[col]
                                        # rrp[f"{col}_smooth"] = rrp[f"{col}_smooth"].rolling(4).mean()
                                    
                                    RRP = rrp30.reindex(rrp.index).fillna(method='bfill') # downsample back to 5mins # WPT.downsample(rrp.copy(),5,30,aggfunc=np.mean)
                                    RRP_actual = RRP[[col for col in RRP.columns if 'obj' not in col]] # this is the actual RRP
                                    RRP = RRP[[col for col in RRP.columns if 'obj' in col]] # this is the objective RRP
                                    
                                    RRP_actual.columns = [col.replace(f"_{region}","") for col in RRP_actual.columns]
                                    RRP.columns = [col.replace(f"_{region}","").replace('_obj',"") for col in RRP.columns]
                                    
                                else:
                                    RRP = rrp30.reindex(rrp.index).fillna(method='bfill') # downsample back to 5mins  # WPT.downsample(rrp.copy(),5,30,aggfunc=np.mean)
                                    RRP.columns = [col.replace(f"_{region}","") for col in Region.Daily[table].columns]
                                    RRP_actual = None
                                
                                
                                # Zero any markets that aren't in the list of stated markets
                                for col in RRP.columns:
                                    if col not in markets:
                                        RRP[col] = 0
                                
                                # set a key containing a bunch of metadata
                                markets_check = [col for col in RRP.columns if RRP[col].sum() != 0]
                                metaVal = createMetaVal2(shorthand,freq,tFcst_str,tInt_str,eta,regDisFrac,regDict_version) # f"{','.join([col for col in RRP.columns if RRP[col].sum() != 0])}_{freq}_{tFcst}_{tInt}_{eta}_{regDisFrac}"
                                
                                # breakpoint()
                                if type(RRP_actual) == pd.core.frame.DataFrame: # if there was a difference between the actual and the objective RRP, save the objective RRP under metaval
                                     # Update existing results object if it exists. Otherwise create it and then update it
                                    try:
                                        Region.RRP_obj[metaVal] = RRP.copy()
                                    except AttributeError:
                                        # no attribute RESULTS
                                        Region.RRP_obj = {metaVal:RRP.copy()} # easy - just declare the variable directly in the correct structure
        
                                # breakpoint()
                                for sMax in sMaxs:
                                    print(sMax)
                                    
                                    st0 = sMax/2 # h
                                    # RRP.dropna(how='all',inplace=True) # drop missing timestamps
                                    # run the results
                                    results = horizonDispatch2(RRP,m,freq,tFcst,tInt,sMax=sMax,st0=st0,eta=eta,rMax=rMax,regDisFrac=regDisFrac,regDict=regDict,debug=True,rrp_actual=RRP_actual)
                                    
                                    # Update existing results object if it exists. Otherwise create it and then update it
                                    try:
                                        Region.RESULTS[metaVal][sMax] = results.copy()
                                    except AttributeError:
                                        # no attribute RESULTS
                                        Region.RESULTS = {metaVal:{sMax:results.copy()}} # easy - just declare the variable directly in the correct structure
                                    except KeyError:
                                        # RESULTS exists, but not under the key metaVal
                                        Region.RESULTS[metaVal] = {sMax:results.copy()} # add the nested dictionary under sMax to the new metaVal key
                                        
                                    m.clear()
                            
                                NEM.pickleNEM()

#%%
# Plot the time-series of the optimisation
year =  2019
NEM = NEMs[year]
shorthand = 'RRP_smooth'
regDisFrac = 0 #  regDisFrac
metaVal = createMetaVal2(shorthand,5,f'all/{div}',f'all/{div}',eta,regDisFrac,regDict_version)
# for year,NEM in NEMs.items():
for region in regions:
    Region = NEM.Regions[region]
    for sMax in sMaxs:
        results = Region.RESULTS[metaVal][sMax]
        secondary_y = [col for col in results.columns if 'RRP' in col]
        Line = [col for col in results.columns if col != 'st_MWh']
        fig = WPT.pandasPlotly2(
            results,
            secondary_y=secondary_y,
            Line=Line,
            Scatter=['st_MWh'],
            Bar=[],
            fill={'st_MWh':'tozeroy'},
            ylabel=f'Discharge rate (MW) & Storage level (MWh)',
            y2label=f'{region} RRP ($/MWh) & Daily profit ($)',
            xlabel='',
            title=f"{rMax}MW, {sMax}h BESS, {region} ({year})",
            exportHTML=os.path.join(myPath,resultsDir,f'{Region.fileName()}_{metaVal.replace(",","_").replace(".","-").replace("/","")}_timeseries.html')
            )
            
#%%
sMax = 2
rMax = 1
markets = ['RRP'] # fields # ['RRP']
shorthand = 'RRP_smooth'
freq = 5
eta = 0.85
regDisFrac = 0
regions = ['SA1']
regDict_version = 'v0'
title_desc = 'arbitrage' #' co-optimised'

# Fig7
# Create a dict of pandas dataframes called annual that stores the annual profit, for each revenue stream and each year, lending to an easy bar chart
annual = {}
for region in regions:
    print(region)
    annual[region] = pd.DataFrame(columns=[f"{field}_Profit_$" for field in fields])
    for year in years:
        print(year)
        NEM = NEMs[year]
        Region = NEM.Regions[region]
        
        # Pull out the length of rrp to get the tFcst and tInt lengths
        rrp = Region.Daily[table]
        # try:
        tFcst,tFcst_str = create_tFcst(len(rrp),hr_frac,div=div) 
        tInt,tInt_str = create_tInt(len(rrp),hr_frac,div=div)
        metaVal = createMetaVal2(shorthand,freq,tFcst_str,tInt_str,eta,regDisFrac,regDict_version)
        results = Region.RESULTS[metaVal][sMax] # get annual results from NEM class
        # except:
        #     tFcst = 2196
        #     tInt = 131760
        #     metaVal = createMetaVal(markets,freq,tFcst,tInt,eta,regDisFrac)
        #     results = Region.RESULTS[metaVal][sMax] # get annual results from NEM class
            
        profitCols = [col for col in results.columns if 'Profit' in col and 'Daily' not in col]
        annual[region].loc[f"{year}/{str(year+1)[2:]}"] = results[profitCols].sum()*rMax
    
    annual[region].columns = [col.split('_')[0][:-3] if col[:3] != 'RRP' else 'Energy' for col in annual[region].columns]

    # # annual[region].plot(kind='bar',stacked=True,colormap='viridis')
    # WPT.pandasPlotly2(
    #     annual[region],
    #     Bar=annual[region].columns,
    #     stacked=True,
    #     ylabel="Annual Profit ($)",
    #     xlabel="Financial Year",
    #     title=f"Annual {title_desc} profit per NEM market: {region[:-1]}, {rMax}MW, {sMax}h storage",
    #     exportHTML=os.path.join(myPath,resultsDir,f"annual_{shorthand}_profit_{region}_{rMax}_{sMax}h_breakdown.html"),
    #     exportPNG=os.path.join(myPath,resultsDir,f"annual_{shorthand}_profit_{region}_{rMax}_{sMax}h_breakdown.png"),
    #     colorList=plotly.colors.qualitative.Antique
    #     )

#%%
# Compare the co-optimised vs pure arbitrage results on a single plot
regions = ['NSW1','SA1']
shorthand = 'FCASCoopt'
regDict_version = 'v0'
regDisFrac = 0.2
freq=5
eta=1
hr_frac = freq/60
div=4
legend_traceorder = 'reversed+grouped' # 'normal'
title_desc=  'Co-optimised'
order = ['Energy (Pure Arbitrage)','Energy','LOWERREG','RAISEREG','LOWER5MIN','LOWER60SEC','LOWER6SEC','RAISE5MIN','RAISE60SEC','RAISE6SEC']

annual = {}
annual_arb = {}
annual_comb = {}
for region in regions:
    print(region)
    annual[region] = pd.DataFrame(columns=[f"{field}_Profit_$" for field in fields])
    annual_arb[region] = annual[region].copy()
    for year in years:
        print(year)
        NEM = NEMs[year]
        Region = NEM.Regions[region]
        
        # Pull out the length of rrp to get the tFcst and tInt lengths
        rrp = Region.Daily[table]

        tFcst,tFcst_str = create_tFcst(len(rrp),hr_frac,div=div) 
        tInt,tInt_str = create_tInt(len(rrp),hr_frac,div=div)
        metaVal = createMetaVal2('RRP',freq,tFcst_str,tInt_str,eta,0,regDict_version)
        results_arb = Region.RESULTS[metaVal][sMax] # get annual results from NEM class

        metaVal = createMetaVal2(shorthand,freq,tFcst_str,tInt_str,eta,regDisFrac,regDict_version)
        results = Region.RESULTS[metaVal][sMax] # get annual results from NEM class
            
        profitCols = [col for col in results.columns if 'Profit' in col and 'Daily' not in col]
        annual[region].loc[f"{int(year)}/{str(int(year+1))[2:]}"] = results[profitCols].sum()*rMax
        
        # create a separate dataframe for custom set of markets
        annual_arb[region].loc[f"{int(year)}/{str(int(year+1))[2:]}"] = results_arb[profitCols].sum()*rMax
        # remove the columns that sum to 0
        annual_arb[region] = annual_arb[region][[f"{col}" for col in annual_arb[region].columns if  annual_arb[region][col].sum() != 0]]
    
    # Flag the arbitrage dataframe with arbitrage keyword
    annual_arb[region].columns = [col.split('_')[0][:-3] if col[:3] != 'RRP' else 'Energy' for col in annual_arb[region].columns]    
    annual_arb[region].columns = [f"{col} (Pure Arbitrage)" for col in annual_arb[region].columns]
 
    annual[region].columns = [col.split('_')[0][:-3] if col[:3] != 'RRP' else 'Energy' for col in annual[region].columns]
    
     # Combine the arbitrage and co-optimised 
    annual_comb[region] = pd.concat([annual[region],annual_arb[region]],axis=1)


    WPT.pandasPlotly2(
        annual_comb[region],
        Bar=[col for col in order if col in annual[region].columns],
        Scatter = annual_arb[region].columns,
        mode='markers',
        ms=15,
        stacked=False,
        relative=True,
        ylabel="Annual Profit ($)",
        xlabel="Financial Year",
        title=f"Annual {title_desc} profit per NEM market: {region[:-1]}, {rMax}MW, {sMax}h storage",
        exportHTML=os.path.join(myPath,resultsDir,f"annual_{shorthand}_profit_{region}_{rMax}_{sMax}h_breakdown.html"),
        exportPNG=os.path.join(myPath,resultsDir,f"annual_{shorthand}_profit_{region}_{rMax}_{sMax}h_breakdown.png"),
        colorList=plotly.colors.qualitative.Antique,
        legend_traceorder=legend_traceorder,
        height=600,
        width=1600,
        ylim=[-50e3,2.6e6]
        )

#%%
# Figure 8a - probability of exceedence chart
sMax = 2

sortDPs = {region: pd.DataFrame() for region in regions}

markets = ['RRP','RAISEREGRRP','LOWERREGRRP','RAISE5MINRRP','LOWER5MINRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE6SECRRP','LOWER6SECRRP']

for market in markets:
    for region in regions:
        for year,NEM in NEMs.items():
            DailyProfit = NEM.Regions[region].results[sMax][f'{market}_DailyProfit_$'] # Series of Daily profit in freq
            # Need to resample to a Daily freq
            # print(DailyProfit)
            DailyProfit = DailyProfit.shift(-1,freq=f"{freq}T") # Shift back by one ts
            DailyProfit = DailyProfit.resample('d').mean()
            # print(DailyProfit)
            # WPT.quickCheck()
            
            # Turn DailyProfit time-series into a probability of exceedence series
            sortDP = DailyProfit.sort_values()[::-1]
            
            try:
                exceedenceDP = 100*np.arange(1.,365+1) / 365
                sortDP.index = exceedenceDP
            except ValueError:
                exceedenceDP = 100*np.arange(1.,366+1) / 366
                sortDP.index = exceedenceDP
            
            # Add to a list of pandas Series
            sortDPs[region][year] = sortDP
        
        ax = sortDPs[region].plot(title=f'Probability of exceedence curves for {market} daily revenue ({region})',logy=True,ylim=(10,10**5))
        ax.set_xlabel('Probability of exceedence (%)')
        ax.set_ylabel('Daily Revenue ($/MW)')
        plt.show()
    
#%%
# Pure arbitrage price bands analysis
regDisFrac = 0
regDict_version = 'v0'
priceBands = [(0,100),(100,300),(300,1000),(1000,10000),(10000,16000)]
rMax = 1
sMax= 2
eta=1
quarterly=False
regions = ['QLD1','SA1','TAS1','VIC1','NSW1']
export = True
shorthand = 'RRP'
years = NEMs.keys()

if quarterly:
    mode="quarterly"
else:
    mode="annual"        


pcBD = {}
for region in regions:
    pCols = [f"{dP}_profit" for dP in priceBands] 
    pCols.extend(['Price Bands ($/MWh)']) # dummy legend
    cCols = [f"{dP}_count" for dP in priceBands] 
    cCols.extend(['Day Count']) # dummy legend
    pBD = pd.DataFrame(columns=pCols) # annual profit breakdown by daily price band
    cBD = pd.DataFrame(columns=cCols) # annual count of daily price delta bands
    
    for year in years:
        NEM = NEMs[year]
        
        Region = NEM.Regions[region] # pull out the region object that contains the raw results
        
        # Pull out the length of rrp to get the tFcst and tInt lengths
        rrp = Region.Daily[table]

        tFcst,tFcst_str = create_tFcst(len(rrp),hr_frac,div=div) 
        tInt,tInt_str = create_tInt(len(rrp),hr_frac,div=div)
        metaVal = createMetaVal2(shorthand,freq,tFcst_str,tInt_str,eta,regDisFrac,regDict_version)
        
        dailyPriceBands(Region,metaVal,priceBands,sMax,freq=freq,mode=mode) # create Region.dpPres
        
        # print(Region.dpPres)
        # breakpoint()
                                
        for key,dPDict in Region.dpPres.items():
            if mode == 'annual':
                name = f"{key.year}-{str(key.year + 1)[2:]}"
            else:
                name = key
            for dP,dailyRes in dPDict.items():                   
                pBD.loc[str(name),f"{dP}_profit"] = dailyRes[r'Profit_$/MWh'].sum()*rMax # take the sum of the annual profit in each slice
                cBD.loc[str(name),f"{dP}_count"] = dailyRes[r'Price_delta_$/MWh'].count()# take the count of rows in each daily price delta slice
                if cBD.loc[str(name),f"{dP}_count"] == 0:
                    cBD.loc[str(name),f"{dP}_count"] = np.nan
            

    pcBD[region] = pd.concat([pBD,cBD],axis=1).sort_index() # concatenate count and profit for convenience


    
#%%
if export:
    pcBD_export = pd.DataFrame()
    for region,pcBD_reg in pcBD.items():
        pcBD_reg_exp = pcBD_reg.copy()
        pcBD_reg_exp['region'] = region
        pcBD_export = pcBD_export.append(pcBD_reg_exp)
    pcBD_export.to_csv(os.path.join(myPath,resultsDir,'pcBD.csv'))

#%%
pcBD_tot = pcBD_export.groupby('region').sum()
fig = WPT.pandasPlotly2(
    pcBD_tot,
    Bar=[col for col in pcBD_tot if pcBD_tot[col].sum() != 0],
    secondary_y = [col for col in pcBD_tot if 'profit' in col],
    stacked=True
    )
#%%
colorList = copy.deepcopy(plotly.colors.qualitative.Antique)
colorList[len(priceBands)] = 'rgb(128,128,128,0)' # grey out the color corresponding to the legend
for region,profitBD in pcBD.items():
    Bar = list(pBD.columns)
    Line = list(cBD.columns)
    fig = WPT.pandasPlotly3(
        pcBD[region],
        Bar=Bar,
        Line=Line,
        stacked=True,
        secondary_y=Line,
        showLegDict={col:True if 'profit' in col or 'Price' in col or 'Count' in col else False for col in pcBD[region].columns},
        ylabel="Annual Revenue ($)",
        y2label="# of days",
        xlabel="Financial Year",
        title=f"{region[:-1]}, {rMax}MW, {sMax}h",
        labels={col:col.split('_')[0].replace('(','').replace(')','').replace(', ','-') for col in pcBD[region].columns},
        exportHTML=os.path.join(myPath,resultsDir,f"{region}_{sMax}_breakdown_{mode}.html"),
        exportPNG=os.path.join(myPath,resultsDir,f"{region}_{sMax}_breakdown_{mode}.png"),
        y2Type='log',
        xType='category',
        ms=30,
        lw=5,
        mode='markers',
        fontsize=25,
        # symbol="triangle-up",
        colorList=colorList,
        legendDict={'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1}
        )
    
#%%
intvl = 30
priceBands = [(0,30),(30,100),(100,300),(300,1000),(1000,15000)]
markets = ['RAISEREGRRP','RAISE5MINRRP','RAISE60SECRRP','RAISE6SECRRP']


profitBreakDown = {}
for market in markets:
    profitBreakDown[market] = {}
    for region in regions:
        pBD = pd.DataFrame(columns=[f"{dP}_profit" for dP in priceBands]) # annual profit breakdown by daily price band
        cBD = pd.DataFrame(columns=[f"{dP}_count" for dP in priceBands]) # annual count of daily price delta bands
        
        for year,NEM in NEMs.items():
            
            Region = NEM.Regions[region] # pull out the region object that contains the raw results
            
            intervalPriceBands(Region,priceBands,sMax,market=market,freq=freq,intvl=intvl,mode=mode)
            
            # print(Region.dpPres)
            # breakpoint()
                                    
            for key,dPDict in Region.filtPriceRes.items():
                for dP,procRes in dPDict.items():                   
                    pBD.loc[key,f"{dP}_profit"] = procRes[r'Profit_$/MWh'].sum() # take the sum of the annual profit in each slice
                    cBD.loc[key,f"{dP}_count"] = procRes[market].count()# take the count of rows in each daily price delta slice
                    if cBD.loc[key,f"{dP}_count"] == 0:
                        cBD.loc[key,f"{dP}_count"] = np.nan
                
    
                profitBreakDown[market][region] = pd.concat([pBD,cBD],axis=1).sort_index() # concatenate count and profit for convenience

#%%
for market in markets:
    profBD = profitBreakDown[market]
    for region,profitBD in profBD.items():
        Bar = list(pBD.columns)
        Line = list(cBD.columns)
        WPT.pandasPlotly2(
            profBD[region],
            Bar=Bar,
            Line=Line,
            stacked=True,
            secondary_y=Line,
            ylabel="Annual Profit ($)",
            y2label="# of intervals",
            xlabel="Financial Year",
            title=f"{market}: {region}, {sMax}h of storage",
            labels={col:col.split('_')[0] for col in profBD[region].columns},
            exportHTML=os.path.join(myPath,resultsDir,f"{market}_{region}_{sMax}_breakdown_{mode}.html"),
            log2y=True,
            ms=10,
            lw=5,
            mode='lines+markers'
            )

#%%
# plot results for different regDisFrac
regDisFracs = [0,0.05,0.1,0.2,0.3,0.5]
year =  2017
NEM = NEMs[year]
fields = ['RRP','RAISEREGRRP','LOWERREGRRP']
region = 'SA1'
shorthand = f'RegCoopt'
sMax=2

# try looking at:
    # reg raise profit when reg raise price < $50 but energy price > $300
    # reg lower profit when reg lower price > $300 and energy price > $300
lowThresh = 50
highThresh = 300

profitCols = [f"{field}_Profit_$" for field in fields]
profit = []
# cycles = pd.Series(index=regDisFracs,name='Cycles')
# metaVal0 = createMetaVal(fields,5,f'all/{div}',f'all/{div}',eta,0)

# results0 = Region.RESULTS[metaVal0][sMax]
# cycles0 = (hr_frac/sMax)*results0['dt_MW'].abs().sum()
# profit0 = results0[profitCols].sum()

Region = NEM.Regions[region]
for regDisFrac in regDisFracs:
    metaVal = createMetaVal2(shorthand,5,f'all/{div}',f'all/{div}',eta,regDisFrac,regDict_version)
    results = Region.RESULTS[metaVal][sMax].copy()
    filt_results = results.copy()[['RAISEREGRRP_Profit_$','LOWERREGRRP_Profit_$','RRP_Profit_$']] # create a dummy variable to hold my filtered tseries
    
    # reg raise profit when reg raise price < $50 but energy price > $300
    filt_raise = (results['RAISEREGRRP'] > lowThresh) | (results['RRP'] < highThresh)
    filt_results.loc[filt_raise,'RAISEREGRRP_Profit_$'] = 0
    # reg lower profit when reg lower price > $300 and energy price > $300
    filt_lower = (results['LOWERREGRRP'] < highThresh) | (results['RRP'] < highThresh)
    filt_results.loc[filt_lower,'LOWERREGRRP_Profit_$'] = 0
    
    # Modify so we have lower and raise-filtered RRP columns to now fill
    filt_results = filt_results.rename({'RRP_Profit_$':'raiseFilt_RRP_Profit_$'},axis=1)
    filt_results['lowerFilt_RRP_Profit_$'] = filt_results['raiseFilt_RRP_Profit_$']
    
    # energy profit in each of these cases
    filt_results.loc[filt_raise,'raiseFilt_RRP_Profit_$'] = 0
    filt_results.loc[filt_lower,'lowerFilt_RRP_Profit_$'] = 0
    
    profit.append(filt_results.sum().to_dict())
profit = pd.DataFrame(profit,index=regDisFracs)
    # cycles.loc[regDisFrac] = ((hr_frac/sMax)*results['dt_MW'].abs().sum()-cycles0)/cycles0
# profit['Total_$'] = profit.sum(axis=1)
# toPlot = pd.concat([profit,cycles],axis=1)
#%%
saveStr = os.path.join(myPath,resultsDir,f"{shorthand}_{region}_{sMax}_regDisFrac")
WPT.pandasPlotly2(
            profit,
            Bar=profit.columns,
            # stacked=True,
            # Scatter=['Cycles'],
            # secondary_y=['Cycles'],
            ylabel="Annual Profit ($)",
            # y2label="Increase in cycles (%)",
            xlabel="Regulation dispatch fraction",
            title=f"{shorthand}: {region}, {sMax}h of storage",
            # labels={col:col.split('_')[0] for col in profBD[region].columns},
            exportHTML= saveStr + '.html',
            exportPNG = saveStr + '.png',
            height=800,
            width=1200,
            ms=20,
            lw=5,
            mode='lines+markers',
            colorList=plotly.colors.qualitative.Antique
            )


#%%
# plot results for a given year in different regions for different metaVals. Stacked bar by market. Separate plot for each region
year = 2017
regions = ['SA1']
shorthandDict = {
    'RRP':['RRP'],
    'RegCoopt': ['RRP','REGRAISERRP','REGLOWERRRP'],
    'RegCooptFr': ['RRP','REGRAISERRP','REGLOWERRRP'],
    'RegRaiseCoopt': ['RRP','REGRAISERRP'],
    'RegLowerCoopt': ['RRP','REGLOWERRRP'],
    'ContCoopt': ['RRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP','LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP'],
    'ContRaiseCoopt': ['RRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP'],
    'ContLowerCoopt': ['RRP','LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP']
    }

shorthands = ['RegCooptFr','RegCoopt']
regDisFracs = [0.2]
regDict_versions = ['v1','fr00','fr01','fr02','fr03','fr04','fr05']
etas = [0.85]
rMax = 1
smax = 2
fields = [
    'RRP',
    'RAISE6SECRRP',
    'RAISE60SECRRP',
    'RAISE5MINRRP',
    'RAISEREGRRP',
    'LOWER6SECRRP',
    'LOWER60SECRRP',
    'LOWER5MINRRP',
    'LOWERREGRRP'
    ]

indexName_mvPos = -1 # position in metaval that you want to use to set the index for annData

title_desc = 'Regulation dispatch fraction'
ylim = [0,1600000]

# build list of metavals
metavals = []
for shorthand in shorthands:
    for regDisFrac in regDisFracs:
        for regDict_version in regDict_versions:
            for eta in etas:
                metavals.append(createMetaVal2(shorthand,5,f'all/{div}',f'all/{div}',eta,regDisFrac,regDict_version))  

annDataList = []
regDict = {1:{}}
count = 0
# build up annual revenue dataframe
for region in regions:
    count += 1
    Region = NEMs[year].Regions[region] # pull out Region for the given year
    # instantiate dataframe
    annData = pd.DataFrame(columns=[f"{field}_Profit_$" for field in fields]) 
    for metaval in metavals:
        # pull out results
        try:
            results = Region.RESULTS[metaval][smax]
        except KeyError:
            print(f"Couldn't find metaval: {metaval}")
            continue
        
        # pull out the list of markets corresponding to eachshorthand key
        shorthand = metaval.split('_')[0] 
        markets = shorthandDict[shorthand] 
               
        # pull out the columns of results that we are interested in (i.e. raw profits)
        profitCols = [col for col in results.columns if ('Profit' in col) and ('Daily' not in col)]
        
        # sum and dump into annData with metaval as the index
        annData.loc[metaval.split('_')[indexName_mvPos].replace('Coopt','').replace('RRP','Arbitrage')] = results[profitCols].sum()*rMax
    
    # breakpoint()
    cleanCols = [col for col in annData.columns if annData[col].sum() != 0]
    annData = annData[cleanCols]
    annData.columns = [col.split('_')[0][:-3] if col[:3] != 'RRP' else 'Energy' for col in annData.columns] # rename the columns
    
    annData.columns = [f"{col}_{region}" for col in annData.columns]
    regDict[1][count] = list(annData.columns)
    annDataList.append(annData)

annData = pd.concat(annDataList,axis=1)

saveStr = os.path.join(myPath,resultsDir,f"{year}_{shorthand}_{title_desc}_{smax}h")
# need to convert this into a subplot!!!
labels = {col:col.split('_')[0] for col in annData.columns}
showLegDict = {col:True if regions[0] in col else False for col in annData.columns}
WPT.pandasPlotly3(
    annData,
    Bar=regDict,
    labels=labels,
    stacked=True,
    subtitles=regions,
    showLegDict=showLegDict,
    ylabel={1:{1:"Annual Profit ($)",2:''}},
    ylim={1:{1:ylim,2:ylim}},
    xlabel='',
    title=f"Annual profits for different {title_desc} co-optimisations: {rMax}MW, {sMax}h storage",
    exportHTML=saveStr + '.html',
    exportPNG=saveStr + '.png',
    colorList=plotly.colors.qualitative.Antique,
    rows=1,
    cols=len(regions),
    fontsize=20,
    height=600,
    width=1400,
    xType='category'
    )

#%%
# def cycles(st,smax,window=False):
#     """
#     Takes a pandas series of the state of charge of a battery at a given time, t, as well as the storage capacity of the battery.
#     Returns the cumulate number of cycles within the given period and the min and max state of charge.
    
#     Args:
#         st (pandas Series): Index should be a datetime index, but not required. Units of st and smax must match, but any unit will work. Hours is recommended.
        
#         smax (float): The storage capacity of the battery. Hours is the recommended unit, but any unit will do, as long as smax and st are in the same unit
        
#         window (bool or int. Default=False): If False, does nothing. Otherwise, this is the rolling average window to be applied to st to smooth out micro
#             optimisations.
        
#     Returns:
#         num_cycles (float): Number of cycles across the time period.
        
#         minSoC (float): Manimum SoC as a ratio of smax.
        
#         maxSoC (float): Maximum SoC as a ratio of smax.
        
#     Created on 30/9/2020 by Bennett Schneider
    
#     """
#     if window:
#         dt_eff = st.rolling(window).mean().diff().abs() # absolute difference between successive states of charge in hours
#     else:
#         dt_eff = st.diff().abs() # absolute difference between successive states of charge in hours
#     num_cycles = dt_eff.sum()/(2*smax) # add all the hours up over the given period and divide by twice the capacity of the battery (cycle is up and down)
#     # min and max SoC over the given period
#     minSoC = st.min()/smax
#     maxSoC = st.max()/smax 
#     return num_cycles,minSoC,maxSoC

# def stackMetaval(df,metaval):
#     """
#     Takes a pandas dataframe associated with a given metaval and assigns columns matching the names of each of the metaval elements and 
#     filled with the value of that metaval element.
#     """
#     df = df.copy()
#     mvList = metaval.split('_')
#     metaDict = {
#         'shorthand': 0,
#         'freq': 1,
#         'H': 2,
#         't': 3,
#         'eta': 4,
#         'Fr': 5,
#         'Fr_obj': 6
#     }
#     for key,val in metaDict.items():
#         df[key] = mvList[val]

#     return df

# def dictFilt(df,selectDict):
#     """
#     Filters a dataframe by a set of criteria stipulated in selectDict.
    
#     Args:
#         df (pandas DataFrame): Any pandas dataframe.
        
#         selectDict (dict of iterables): Keys are column names for df. Values must be iterables. Will filter df based on
#             whether the columns in the keys contain the values in the values of selectDict. If a value is entered in selectDict as None,
#               it will not be used to filter df. If df does not contain a given key in selectDict in its columns, a warning is printed.
            
#     Returns:
#         df_filt (pandas DataFrame): df, but with rows filtered according to selectDict
#     """
#     df_filt = df.copy()
#     for key,val in selectDict.items():
#         if val:
#             try:
#                 df_filt = df_filt[df_filt[key].isin(val)]
#             except KeyError:
#                 WPT.errFunc(f'WARNING: {key} not in df',handled=True)
#             except TypeError:
#                 WPT.errFunc(f'key: {key}\nval:{val}')
    
#     return df_filt
#%% Histogram of the number of cycles in different scenarios across a year
# Create a system for looping through all manner of scenarios and presenting the results per scenario in a series of sub-plots
# https://plotly.com/python/builtin-colorscales/
fcas_type = 'Reg'

# shorthands = ['RRP',f'{fcas_type}Coopt',f'{fcas_type}RaiseCoopt',f'{fcas_type}LowerCoopt'] # 'RegCoopt','RegRaiseCoopt','RegLowerCoopt','ContCoopt','ContRaiseCoopt','ContLowerCoopt'] #,'RaiseRegCoopt','LowerRegCoopt','RegCoopt','RaiseContCoopt','LowerContCoopt','ContCoopt'] #['RRP','RegCoopt','RegRaiseCoopt','RegLowerCoopt']
shorthands= ['RRP'] # [f'{fcas_type}Coopt']
smooths = [False,True]
thresh=40
hrs = 4
roll=False


regDisFracs = [0]  # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] #,0.2]
regDict_versions = ['v0'] # ['fr00','fr01','fr02','fr03','fr04','fr05','fr06','fr07','fr08','fr09','fr10'] # ['v0'] # ['v0','v1','fr00','fr01','fr02','fr03','fr04','fr05']
etas = [1]
Hs = [f"all/4"]  #  [f"{h}h" for h in [1,2,4,8,16]]  # 
ts = [f"all/4"] # ['30m'] 
freqs = [5]

years = [2019] # [2006,2012,2019] # list(np.linspace(2006,2019,14)) # [2006,2012,2019] #
regions = ['NSW1']  #,'NSW1','TAS1','QLD1','VIC1'] # 'VIC1','QLD1','TAS1','SA1'] # ['SA1','NSW1'] # 
smaxs = [2] # [2,4,8]

autoCyc = True
minCyc = 0
maxCyc = 8
windows = [False] #[False] #[False,2*12] # [False,2*12] #[6,3,False] # 6 is 30mins

pBands = [0,100,300,1000,10000,16000]

# build list of metavals
metavals = []
for smooth in smooths:
    if smooth:
        shorthands_smooth = [f"{sh}-{thresh}-{hrs}-{roll}" for sh in shorthands]
    else:
        shorthands_smooth = shorthands
    for shorthand in shorthands_smooth:
        for regDisFrac in regDisFracs:
            for regDict_version in regDict_versions:
                for eta in etas:
                    for H in Hs:
                        for t in ts:
                            for freq in freqs:
                                metavals.append(createMetaVal2(shorthand,freq,H,t,eta,regDisFrac,regDict_version))  
                                
#%%

dailyDF_all = []
histDF_all = []
pHistDF_all = []
resultsDF_all = []
rrp_obj_all = []
pBandsDF_all = []
# Loop through all the possible result permutations
for region in regions:
    print(region)
    for year in years:
        print(year)
        Region = NEMs[year].Regions[region]
        # Region.cycles = {'num':{},'soc_min':{},'soc_max':{}}
            
        for metaval in metavals:
            if '-' in metaval:
                smooth = True
            else:
                smooth = False
                
            # pull out results
            try:
                resultsDict = Region.RESULTS[metaval]
            except KeyError:
                print(f"Couldn't find results for metaval: {metaval}")
                continue
            
            try:
                rrp_obj = Region.RRP_obj[metaval].copy()
                rrp_obj['FY'] = str(int(year))
                rrp_obj['region'] = region
            except KeyError:
                print(f"Couldn't find objective rrp for metaval: {metaval}")
                rrp_obj = pd.DataFrame()
            except AttributeError:
                print(f"{region} {year} doesn't have an RRP_obj")
                rrp_obj = pd.DataFrame()
                
            
            # Region.cycles.update({key:{metaval:})
                
            for smax in smaxs:
                try:
                    results = resultsDict[smax].copy()
                except KeyError:
                    print(f"Couldn't find smax: {smax}")
                    continue
                
                # breakpoint()
                # results['Hour'] = results.index.hour
                # results['Month'] = results.index.month
                # pivot = results.pivot(index='Hour',columns='Month',values='st_MWh')
                # Region.save(pivot,'pivot')
                
                results['smooth'] = str(smooth)
                # add a days teg
                results['Day'] = (results.index - results.index[0]).days
                # add other column tags to results
                results['region'] = region
                results['FY'] = str(int(year))
                
                results = stackMetaval(results,metaval) # do for each element of metaval
                results['smax'] = smax
    
                
                for window in windows:
                    
                    # Generate a daily cycles dataframe
                    days = list(set(list(results['Day'])))
                    dailyDF = pd.DataFrame(
                        [
                            cycles(results[results['Day']==day]['st_MWh'],smax,window=window) for day in days
                            ],
                        index=days,
                        columns=['Cycles','minSoC','maxSoC']
                        )
                    
                    # add column tags to this dataframe
                    dailyDF['region'] = region
                    dailyDF['FY'] = str(int(year))
                    dailyDF = stackMetaval(dailyDF,metaval) # do for each element of metaval
                    dailyDF['smax'] = smax
                    dailyDF['window'] = str(window)
                    dailyDF['smooth'] = str(smooth)
                    
                    # Add daily profit to dailyDF
                    profitCols = [col for col in results.columns if 'Profit' in col and 'Daily' not in col] # define columns to sum
                    profitCols.append('Day')
                    profitDF = results[profitCols].groupby('Day').sum()
                    profitDF.columns = [col.split('_')[0] for col in profitDF.columns]
                    profitDF = profitDF[[col for col in profitDF.columns if profitDF[col].sum() != 0]]
                    
                    dailyDF = pd.concat([dailyDF,profitDF],axis=1) # add to dailyDF
                    
                    # Add daily price bands to dailyDF
                    priceCols = [col for col in results.columns if 'RRP' in col and 'Profit' not in col]
                    for prCol in priceCols: # for each market price
                        # get the minimum and maximum price on a given day
                        dailyDF[f'min{prCol}'] = results[[prCol,'Day']].groupby('Day').min() 
                        dailyDF[f'max{prCol}'] = results[[prCol,'Day']].groupby('Day').max()
                        if prCol == 'RRP':
                            # for RRP, take the difference to get a test for our price band
                            dailyDF[f'del{prCol}'] = dailyDF[f'max{prCol}'] - dailyDF[f'min{prCol}'] 
                    dailyDF['PriceBands'] =  pd.cut(dailyDF[f'delRRP'],pBands) # create price bands
                    
                    # Create a histogram dataframe and do the same for it
                    if autoCyc:
                        minCyc = math.floor(dailyDF['Cycles'].min()*10)/10
                        maxCyc = math.ceil(dailyDF['Cycles'].max()*10)/10
                    counts,bins = np.histogram(dailyDF['Cycles'],bins=np.linspace(minCyc,maxCyc,int((maxCyc-minCyc)*10+1)))
                    # also want profit as a variable in these bins
                    dailyDF['Bins'] = pd.cut(dailyDF['Cycles'],bins,right=False) # create column of dailyDF (a daily tseries) containing bin definitions
                    
                    # Bin each profit category and sum
                    
                    # bin by cycles
                    profitSlice = list(profitDF.columns) 
                    profitSlice.append('Bins')
                    pHistDF = dailyDF[profitSlice].groupby('Bins').sum()
                    bins = 0.5 * (bins[:-1] + bins[1:]) # rename bins based on their centre
                    pHistDF.index = bins # assign to the index
 
                    # bin by price range
                    profitSlice = list(profitDF.columns) 
                    profitSlice.append('PriceBands')
                    pBandsDF = dailyDF[profitSlice].groupby('PriceBands').sum()
                    
                    # stack all the profit levels and reset the index. Name the stacked column "Profit" and the reset index to "Market"
                    pBandsDF = pBandsDF.stack().reset_index(1,name='Profit').rename({'level_1':'Market'},axis=1)
                    pBandsDF['region'] = region
                    pBandsDF['FY'] = str(int(year))
                    pBandsDF = stackMetaval(pBandsDF,metaval) # do for each element of metaval
                    pBandsDF['smax'] = smax
                    pBandsDF['window'] = str(window)
                    pBandsDF['smooth'] = str(smooth)
                
                    # do the same for price range DF
                    pHistDF = pHistDF.stack().reset_index(1,name='Profit').rename({'level_1':'Market'},axis=1)
                    pHistDF['region'] = region
                    pHistDF['FY'] = str(int(year))
                    pHistDF = stackMetaval(pHistDF,metaval) # do for each element of metaval
                    pHistDF['smax'] = smax
                    pHistDF['window'] = str(window)
                    pHistDF['smooth'] = str(smooth)
                    
                         
                    # create histogram for count of cycles
                    histDF = pd.DataFrame(counts,index=bins,columns=['Count'])
                    histDF['region'] = region
                    histDF['FY'] = str(int(year))
                    histDF = stackMetaval(histDF,metaval) # do for each element of metaval
                    histDF['smax'] = smax
                    histDF['window'] = str(window)
                    histDF['smooth'] = str(smooth)
                    
                    # breakpoint()
                    
                    
                    # breakpoint()
                    # append to the list
                    resultsDF_all.append(results)
                    dailyDF_all.append(dailyDF)
                    histDF_all.append(histDF)
                    pHistDF_all.append(pHistDF)
                    pBandsDF_all.append(pBandsDF)
            rrp_obj_all.append(rrp_obj)

# concatenate vertically
resultsDF_all = pd.concat(resultsDF_all)
dailyDF_all = pd.concat(dailyDF_all) 
histDF_all = pd.concat(histDF_all)
pHistDF_all = pd.concat(pHistDF_all)
rrp_obj_all = pd.concat(rrp_obj_all)
pBandsDF_all = pd.concat(pBandsDF_all)

histDF_all['Daily Cycles'] = histDF_all.index
pHistDF_all['Daily Cycles'] = pHistDF_all.index

#%% Cycles histogram
# Settings
facet_row = 'Fr' # 'shorthand'
facet_col = None # None
color = 'shorthand'
barmode = 'stack' # 'group', 'stack'
fcas_type = 'Reg'
select = {} # {'region':'NSW1','FY':'2019'} # name of column not differentiated, but contained as an option in histDF_all
agg = 'Daily Cycles'
name = 'Fr' # f'{fcas_type}Coopt_2019_NSW' # 'Arbitrage_annual_regional'
width = 1000
height = 600
lorient = 'h'
legend_traceorder='reversed'
hideAnnotations = True


category_orders = {} # {'shorthand':['Arbitrage','Cont','ContLower','ContRaise']}
label_content = {sh:sh.replace('Coopt','') for sh in set(pHistDF_all['shorthand'].copy())}
label_content.update({mk:mk.replace('RRP','') if mk != 'RRP' else mk.replace('RRP','Arbitrage') for mk in set(pHistDF_all['Market'].copy())})
label_content.update({sh:sh.split('-')[0].replace('RRP','Smoothed Arbitrage')  for sh in set(pHistDF_all['shorthand'].copy()) if '-' in sh})

orderColor = [0,3,2,1,2,5,6,7,8,9,10] # [2,1,0,3,4,5,6,7,8,9,10] # [2,1,0,3,2,5,6,7,8,9,10]
removeColor = False # [1,1,1]
category_orders = {} # {'Market':['RegRaise>RegLower','RegLower>RegRaise']}


saveStr = os.path.join(myPath,resultsDir,f"cyclesHistogram_{name}"  + '_'.join([f"{key}{val}" for key,val in select.items()]))

# We may not have covered all the options at hand. This lets us remove the options not selected so we don't aggregate without categorising               
histDF_all_plot = histDF_all.copy()
for key,val in select.items():
    histDF_all_plot = histDF_all_plot[histDF_all_plot[key] == val]
    
for col in ['shorthand']:
    histDF_all_plot[col] = histDF_all_plot[col].map(label_content)

colorList = copy.deepcopy(plotly.colors.qualitative.Antique)
if orderColor:
    colorList = [colorList[order] for order in orderColor]
if removeColor:
    for rc in removeColor:
        colorList.remove(colorList[rc])

fig = px.bar(
    histDF_all_plot,
    x='Daily Cycles',
    y='Count',
    facet_row=facet_row,
    facet_col=facet_col,
    color=color,
    color_discrete_sequence=colorList,
    category_orders=category_orders
    ).for_each_trace(lambda t: t.update(name=t.name.split('=')[-1]))

if hideAnnotations:
    for anno in fig['layout']['annotations']:
        anno['text']=''

if lorient == 'h':
    legendDict = {'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1}
else:
    legendDict = None
    
fig.update_layout(font={'size':25},legend_traceorder=legend_traceorder,barmode=barmode,legend=legendDict)
# fig = px.histogram(dailyDF_all,x='Cycles')    
     
plot(fig,filename=saveStr + '.html')
fig.write_image(saveStr + '.png',width=width,height=height)

#%% Profit per cycle bin bar plot
# Settings
x = 'H' # 'Daily Cycles'
y = 'Profit'
fcas_type = 'Reg'
facet_row = None #'region'
facet_col = 'smax'  #'window' # None
color = None
barmode = 'group' # 'group', 'stack'
select =  {} # {'window':'False'} # name of column not differentiated, but contained as an option in pHistDF_all
name = 'Arbitrage_ForecastHorizon' # f'Coopt_{fcas_type}' # 'Arbitrage_annual_regional'
legend_traceorder = 'reversed'
width = 1200
height = 500
lorient = 'v'
try:
    sorted_h = [float(h.replace('h','')) for h in set(list(pHistDF_all['H']))]
    sorted_h.sort()
    sorted_h = [f"{h}h".replace('.0','') for h in sorted_h]
except KeyError:
    pass
category_orders = {
    'shorthand':['Arbitrage',fcas_type,f'{fcas_type}Lower',f'{fcas_type}Raise'],
    'Market':['Arbitrage',fcas_type,f'{fcas_type}Lower',f'{fcas_type}Raise'],
    'H':sorted_h
    }
label_content = {sh:sh.replace('Coopt','') for sh in set(pHistDF_all['shorthand'].copy())}
label_content.update({mk:mk.replace('RRP','') if mk != 'RRP' else mk.replace('RRP','Arbitrage') for mk in set(pHistDF_all['Market'].copy())})
labels = {'shorthand':'Scenario','Profit': 'Annual Revenue ($)'}

saveStr = os.path.join(myPath,resultsDir,f"{name}_{x}_{y}_{facet_row}_{facet_col}_{color}" + '_'.join([f"{key}{val}" for key,val in select.items()]))

# We may not have covered all the options at hand. This lets us remove the options not selected so we don't aggregate without categorising               
pHistDF_all_plot = pHistDF_all.copy()

for key,val in select.items():
    pHistDF_all_plot = pHistDF_all_plot[pHistDF_all_plot[key] == val]
groupList = [x,facet_row,facet_col,color]


try:
    lg = len(groupList)
    for i in range(lg):
        groupList.remove(None)
    # groupList.remove(None)
except ValueError:
    pass
pHistDF_all_plot = pHistDF_all_plot.groupby(groupList).sum().reset_index()

for col in ['shorthand','Market']:
    try:
        pHistDF_all_plot[col] = pHistDF_all_plot[col].map(label_content)
    except KeyError:
        pass

# WPT.funDataSaver(pHistDF_all_plot,saveStr + '.csv')

fig = px.bar(
    pHistDF_all_plot.copy(),
    x=x,
    y=y,
    labels=labels,
    facet_row=facet_row,
    facet_col=facet_col,
    color=color,
    color_discrete_sequence=plotly.colors.qualitative.Antique,
    color_continuous_scale=plotly.colors.sequential.thermal,
    barmode=barmode,
    category_orders=category_orders,
    height=height,
    width=width
    ).for_each_trace(lambda t: t.update(name=t.name.split('=')[-1]))

if lorient == 'h':
    legendDict = {'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1}
else:
    legendDict = None
    
fig.update_layout(font={'size':22},legend_traceorder=legend_traceorder,legend=legendDict)
# fig = px.histogram(dailyDF_all,x='Cycles')    
     
plot(fig,filename=saveStr + '.html')
fig.write_image(saveStr + '.png',width=width,height=height)

#%% Profit per price band bin bar plot
# Settings
x = 'shorthand' 
y = 'Profit'
fcas_type = 'Reg'
facet_row = None #'region'
facet_col = None  #'window' # None
color = 'PriceBands'
barmode = 'stack' # 'group', 'stack'
select =  {} # {'window':'False'} # name of column not differentiated, but contained as an option in pHistDF_all
name = 'smooth_vs_micro' # 'Arbitrage_ForecastHorizon' # f'Coopt_{fcas_type}' # 'Arbitrage_annual_regional'
legend_traceorder = 'normal'
category_orders={}
width = 900
height = 400
lorient = 'v'
try:
    sorted_h = [float(h.replace('h','')) for h in set(list(pBandsDF_all['H']))]
    sorted_h.sort()
    sorted_h = [f"{h}h".replace('.0','') for h in sorted_h]
except KeyError:
    pass
except ValueError:
    pass
# category_orders = {
#     'shorthand':['Arbitrage',fcas_type,f'{fcas_type}Lower',f'{fcas_type}Raise'],
#     'Market':['Arbitrage',fcas_type,f'{fcas_type}Lower',f'{fcas_type}Raise'],
#     'H':sorted_h
#     }
label_content = {sh:sh.replace('Coopt','') for sh in set(pBandsDF_all['shorthand'].copy())}
label_content.update({mk:mk.replace('RRP','') if mk != 'RRP' else mk.replace('RRP','Arbitrage') for mk in set(pBandsDF_all['Market'].copy())})
label_content.update({sh:sh.split('-')[0].replace('RRP','Smoothed Arbitrage')  for sh in set(pHistDF_all['shorthand'].copy()) if '-' in sh})
labels = {'shorthand':'Scenario','Profit': 'Annual Revenue ($)','H':'Forecast Horizon (h)'}

saveStr = os.path.join(myPath,resultsDir,f"{name}_{x}_{y}_{facet_row}_{facet_col}_{color}" + '_'.join([f"{key}{val}" for key,val in select.items()]))

# We may not have covered all the options at hand. This lets us remove the options not selected so we don't aggregate without categorising               
dailyDF_all_plot = pBandsDF_all.copy()

for key,val in select.items():
    dailyDF_all_plot = dailyDF_all_plot[dailyDF_all_plot[key] == val]
groupList = [x,facet_row,facet_col,color]


try:
    lg = len(groupList)
    for i in range(lg):
        groupList.remove(None)
    # groupList.remove(None)
except ValueError:
    pass
dailyDF_all_plot = dailyDF_all_plot.groupby(groupList).sum().reset_index()

for col in ['shorthand','Market']:
    try:
        dailyDF_all_plot[col] = dailyDF_all_plot[col].map(label_content)
    except KeyError:
        pass

# WPT.funDataSaver(pHistDF_all_plot,saveStr + '.csv')

fig = px.bar(
    dailyDF_all_plot.copy(),
    x=x,
    y=y,
    labels=labels,
    facet_row=facet_row,
    facet_col=facet_col,
    color=color,
    color_discrete_sequence=plotly.colors.qualitative.Antique,
    color_continuous_scale=plotly.colors.sequential.thermal,
    barmode=barmode,
    category_orders=category_orders,
    height=height,
    width=width
    ).for_each_trace(lambda t: t.update(name=t.name.split('=')[-1]))

if lorient == 'h':
    legendDict = {'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1}
else:
    legendDict = None
    
fig.update_layout(font={'size':22},legend_traceorder=legend_traceorder,legend=legendDict)
# fig = px.histogram(dailyDF_all,x='Cycles')    
     
plot(fig,filename=saveStr + '.html')
fig.write_image(saveStr + '.png',width=width,height=height)

#%% Find day or series of days which meet requirements, slice results accordingly, and plot a time-series of that period in publishable format
selectDict = {
    'shorthand':  ['RegCoopt'], # ['RRP-40-4-False'], # ['RRP'], # ['RRP'],
    'Fr':         ['0.4'],
    'Fr_obj':     None,#['v0'],
    'eta':        ['1'],
    'H':          ["all/4"],
    't':          ["all/4"],
    'freq':       ['5'],
    'FY':         ['2019'], #['2008'], #['2017'],
    'region':     ['NSW1'], #['SA1'], #['SA1'],
    'smax':       [2],
    'window':     ['False'],
    'PriceBands': [pd.Interval(100,300)] # [pd.Interval(10000,16000)]
    }

# Filter dailyDF_all by the selectDict
selectDay = dictFilt(dailyDF_all,selectDict)

print(list(selectDay.index))

#%%
days = [7,8,9] # [210,211,212] #[183, 184, 185]
year = ['2019']
region = ['NSW1']
name = 'NSW_Fr_0-4_100-300' # 'SA_ContRaiseCoopt'
removeColor = False # [1,1,1]
orderColor = False # [0,4,5,6,4,5,6,7,8,9,10]
lineOrder =  ['RRP','RAISEREGRRP','LOWERREGRRP'] #  ['RRP','RRP_obj'] # ['RRP','RAISE60SECRRP','RAISE6SECRRP','RAISE5MINRRP'] # ['RRP','RAISEREGRRP','LOWERREGRRP'] #'LOWER60SECRRP','LOWER6SECRRP','LOWER5MINRRP','RAISE60SECRRP','RAISE6SECRRP','RAISE5MINRRP','RAISEREGRRP','LOWERREGRRP'] # ['RRP','LOWER60SECRRP','LOWER5MINRRP','LOWER6SECRRP','RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP']
selectDict.update({'Day':days,'FY':year,'region':region})

filtResults = dictFilt(resultsDF_all,selectDict).sort_index()

# Create a dataframe showing the cumulative profits for a given day
profitCols =  WPT.getKeys(filtResults.columns,['Profit'],excl=['Daily']) # all the profit columns
cumul_all = filtResults[profitCols].cumsum().sum(axis=1)
cumul_all = cumul_all/cumul_all.max()
cumul_all.name = 'Cumul. rev. (norm)'

# slice out the objective price if it exists
rrp_obj_slice = rrp_obj_all[(rrp_obj_all.index >= filtResults.index.min()) & (rrp_obj_all.index <= filtResults.index.max())]
rrp_obj_slice.columns = [f"{col}_obj" for col in rrp_obj_slice.columns]

toPlot = pd.concat([filtResults,cumul_all,rrp_obj_slice],axis=1)

secondary_y = WPT.getKeys(toPlot.columns,['RRP'],['Cumu'])
Line = WPT.getKeys(toPlot.columns,['RRP'],excl=['Daily','Profit']) # [col for col in toPlot.columns if col != 'st_MWh']
Line = [col for col in Line if toPlot[col].sum() !=0 and 'RRP' in col]

Line = [col for col in lineOrder if col in Line]
Scatter = ['st_MWh','Cumul. rev. (norm)']
Bar = WPT.getKeys(toPlot.columns,['MW'],excl=['MWh','dt_net_MW','regDt'])
Bar = [col for col in Bar if toPlot[col].sum() != 0]
# Scatter.extend(list(cumul_all.columns))
labels = {'dt_MW':'Energy Disp.','st_MWh':'Storage level (h)'}
labels.update({col:col.replace('_Lt_MW',' Lower En.').replace('_Rt_MW','Raise En.') for col in toPlot.columns if col not in labels.keys()})


for col in toPlot.columns:
    if 'Lt' in col:
        toPlot[col] = -toPlot[col]

colorList = copy.deepcopy(plotly.colors.qualitative.Antique)
if orderColor:
    colorList = [colorList[order] for order in orderColor]
if removeColor:
    for rc in removeColor:
        colorList.remove(colorList[rc])

# labels.update({col:f"% Cumul. Profit ({col.split('_')[0]})" for col in cumul_all.columns})
savePath = os.path.join(myPath,resultsDir,f'{name}_Report_timeseries')
fig = WPT.pandasPlotly2(
    toPlot,
    secondary_y=secondary_y,
    Line=Line,
    Scatter=Scatter,
    Bar=Bar,
    labels=labels,
    fill={col:'tozeroy' for col in Scatter},
    ylabel=f'Dispatch (MW), SoC (h), Norm. rev.',
    y2label=f'{selectDict["region"][0]} RRP ($/MWh)',
    xlabel='',
    # title=f'{selectDict["smax"][0]}h BESS, {selectDict["region"][0]} ({selectDict["year"][0]})',
    exportHTML=savePath + '.html',
    exportPNG=savePath + '.png',
    colorList={'Bar':colorList,'Line':colorList,'Scatter':plotly.colors.qualitative.Bold},
    opacity=0.4,
    lw=3,
    legendDict={'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1},
    fontsize=20,
    width=1200,
    height=600,
    log2y=False,
    stacked=False,
    relative=True
    )



print(f"Total Profit: ${toPlot[profitCols].sum().sum():.0f}")
print(f'Total Cycles: {cycles(toPlot["st_MWh"],smax)[0]:.0f}')
#%% Price plot
years = [2006,2012,2019]
regions = ['NSW1','SA1']
freq=120

# Create a stacked version of all the price information, averaged to 30mins and organised by market, financial year, and region
DATA = []
for year in years:
    NEM = NEMs[year]
    print("\n*************")
    print(year)
    print("-----")
    for region in regions:
        Region = NEM.Regions[region]
        print(region)
        data = WPT.timeAvgEnd(Region.Daily[table],5,freq)
        data.columns = [col.replace(f'RRP_{region}','') if col != f'RRP_{region}' else 'RRP' for col in data.columns]
        # breakpoint()
        contCols = [col for col in data.columns if '5' in col or '6' in col]
        data['RAISECONT'] = data[[col for col in contCols if 'RAISE' in col]].sum(axis=1)
        data['LOWERCONT'] = data[[col for col in contCols if 'LOWER' in col]].sum(axis=1)
        data.drop(contCols,axis=1,inplace=True)
        
        data['RegRaise>RegLower'] = data['RAISEREG'][data['RAISEREG']>data['LOWERREG']]
        data['RegLower>RegRaise'] = data['LOWERREG'][data['LOWERREG']>data['RAISEREG']]
        data = data.stack().reset_index(1,name='RRP').rename({'level_1':'Market'},axis=1)
        data['Energy'] = data[data['Market'] == 'RRP']['RRP'] # add the energy market as a separate column
        data = data[data['Market'] != 'RRP']
        data['FY'] = str(year)
        data['region'] = region

        
        # data.rename({'RRP':'FCAS Price ($/MW/hr)'},axis=1,inplace=True)
        # data.rename({'Energy':'Energy Price ($/MWh)'},axis=1,inplace=True)
        DATA.append(data)
    print("*************\n")
DATA = pd.concat(DATA)

#%% Plot in plotly express
x = 'Energy'
y = 'RRP'
color = 'Market'
facet_col = 'FY'
facet_row = 'region'
size = 'Size'
name = 'PriceScatter_Cont'
factors = [1,2,4] # [0.2,0.8,4]
log_x = True
log_y = True
width=1500
height=800
orderColor = [4,1,2,3,5,6,7,8,9,10] # [2,1,0,3,4,5,6,7,8,9,10] # [2,1,0,3,2,5,6,7,8,9,10]
removeColor = False # [1,1,1]
category_orders = {} # {'Market':['RegRaise>RegLower','RegLower>RegRaise']}


markets = ['RAISECONT','LOWERCONT'] # ['RegLower>RegRaise','RegRaise>RegLower']# ['Energy',]

DATA_plot = DATA[DATA['Market'].isin(markets)]
DATA_plot['Size']  = DATA_plot['Energy'].abs() + DATA_plot['RRP'].abs() + 500

labels = {'Energy':'Energy RRP ($/MWh)','RRP':'FCAS RRP ($/MW/hr)'}
lorient = 'v'
legend_traceorder = 'normal' # reversed
fontsize = 25
saveStr = os.path.join(myPath,resultsDir,f"{name}_{x}_{y}_{facet_row}_{facet_col}_{color}_{size}")

colorList = copy.deepcopy(plotly.colors.qualitative.Antique)
if orderColor:
    colorList = [colorList[order] for order in orderColor]
if removeColor:
    for rc in removeColor:
        colorList.remove(colorList[rc])

#%%
fig = px.scatter(
    DATA_plot,
    x=x,
    y=y,
    labels=labels,
    facet_row=facet_row,
    facet_col=facet_col,
    color=color,
    size=size,
    color_discrete_sequence=colorList,
    category_orders=category_orders,
    log_x=log_x,
    log_y=log_y,
    opacity=0.5
    ).for_each_trace(lambda t: t.update(name=t.name.split('=')[-1]))

if lorient == 'h':
    legendDict = {'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1}
else:
    legendDict = None


# fig.update_traces(marker_size=4)
fig.update_layout(
    font={'size':fontsize},
    legend_traceorder=legend_traceorder,
    legend=legendDict,
    xaxis = dict(range=[-0.5,4.8]),
    yaxis = dict(range=[-0.5,4.8]),
    )


rows = len(set(DATA_plot[facet_row]))
cols = len(set(DATA_plot[facet_col]))

for factor in factors:
    name =  f'{factor}'
    for srow in range(1,rows+1):
        for scol in range(1,cols+1):
            print(srow,scol)
            fig.add_trace(
                go.Line(
                    x = np.linspace(1,8000,4),
                    y = factor*np.linspace(1,8000,4),
                    name = name,
                    text=['','','',name],
                    textposition='top center',
                    textfont={'size':12},
                    mode='lines+text',
                    showlegend=False,
                    line = dict(
                        color='rgba(0,0,0,0.5)',
                        width=2,
                    ),
                    ),
                row=srow,
                col=scol
                )         

fig.write_image(saveStr + '.png',width=width,height=height)    
plot(fig,filename=saveStr + '.html')

#%% Probability of exceedence
years = [2019] # np.linspace(2006,2019,14) # [2006,2012,2019]
regions = ['NSW1'] # ['NSW1','SA1']


freq=30

# Create a stacked version of all the price information, averaged to 30mins and organised by market, financial year, and region
PROBEXC = []
for year in years:
    NEM = NEMs[year]
    print("\n*************")
    print(year)
    print("-----")
    for region in regions:
        Region = NEM.Regions[region]
        print(region)
        data = WPT.timeAvgEnd(Region.Daily[table],5,freq)
        data.columns = [col.replace(f'RRP_{region}','') if col != f'RRP_{region}' else 'RRP' for col in data.columns]
        # breakpoint()
        contCols = [col for col in data.columns if '5' in col or '6' in col]
        data['RAISECONT'] = data[[col for col in contCols if 'RAISE' in col]].sum(axis=1)
        data['LOWERCONT'] = data[[col for col in contCols if 'LOWER' in col]].sum(axis=1)
        data.drop(contCols,axis=1,inplace=True)
        data = data.abs()
        probExc = pd.DataFrame()
        for col in data.columns:
            probExc[col] = list(data[col].sort_values()[::-1]) # sort by the particular column
        probExc.index = 100*np.arange(1.,len(probExc)+1) / len(probExc) # assign prob exceedence to index
        # breakpoint()
        probExc = probExc.stack().reset_index(1,name='RRP').rename({'level_1':'Market'},axis=1)

        probExc['FY'] = year
        probExc['region'] = region

        
        # data.rename({'RRP':'FCAS Price ($/MW/hr)'},axis=1,inplace=True)
        # data.rename({'Energy':'Energy Price ($/MWh)'},axis=1,inplace=True)
        PROBEXC.append(probExc)
    print("*************\n")
PROBEXC = pd.concat(PROBEXC)
PROBEXC.index.name = 'ProbExc'
PROBEXC.reset_index(inplace=True)

#%%

y='RRP'
x='ProbExc'
facet_row='Market'
facet_col='region'
size=None
color='FY'
labels={'ProbExc':'Probability of Exceedence (%)','RRP':'FCAS RRP ($/MW/h)'}
log_x=False
log_y=True
orderColor = False # [0,2,1,3,4,5,6,7,8,9,10] # [2,1,0,3,4,5,6,7,8,9,10] # [2,1,0,3,2,5,6,7,8,9,10]
name= ''

saveStr=os.path.join(myPath,resultsDir)
width=1500
height=1800
fs=23
markets = ['RAISEREG','LOWERREG','RAISECONT','LOWERCONT']


colorList = copy.deepcopy(plotly.colors.qualitative.Antique)
if orderColor:
    colorList = [colorList[order] for order in orderColor]

fig = px.scatter(
    PROBEXC[PROBEXC['Market'].isin(markets)],
    x='ProbExc',
    y=y,
    size=size,
    labels=labels,
    facet_row=facet_row,
    facet_col=facet_col,
    color=color,
    color_discrete_sequence=colorList,
    color_continuous_scale=plotly.colors.sequential.RdBu,
    # category_orders=category_orders,
    log_y=log_y,
    log_x=log_x,
    ).for_each_trace(lambda t: t.update(name=t.name.split('=')[-1]))
    
fig.update_layout(
    font={'size':fs},
    xaxis = dict(range=[0,100]),
    yaxis = dict(range=[0,4])
    ) # ,legend_traceorder=legend_traceorder,legend=legendDict)
fig.layout.plot_bgcolor='rgb(80,80,80)'

if saveStr:
    slices = [col for col in [facet_row,facet_col,color,size] if col]
    filepath = os.path.join(saveStr,f"probExceedence_{'_'.join(slices)}")
    plot(fig,filename=filepath + '.html')
    fig.write_image(filepath + '.png',width=width,height=height)

#%% Diurnal box and whisker

year =  2019 # np.linspace(2006,2019,14) # [2006,2012,2019]
region = 'NSW1' #,'SA1'] # ['NSW1','SA1']
freq = 30
diurnal = []
NEM = NEMs[year]
Region = NEM.Regions[region]
data = WPT.timeAvgEnd(Region.Daily[table],5,freq)
data.columns = [col.replace(f'RRP_{region}','') if col != f'RRP_{region}' else 'RRP' for col in data.columns]
# breakpoint()
contCols = [col for col in data.columns if '5' in col or '6' in col]
data['RAISECONT'] = data[[col for col in contCols if 'RAISE' in col]].sum(axis=1)
data['LOWERCONT'] = data[[col for col in contCols if 'LOWER' in col]].sum(axis=1)
data.drop(contCols,axis=1,inplace=True)
data['Hour'] = data.index.hour
data['FY'] = year
data['region'] = region
diurnal.append(data)
diurnal = pd.concat(diurnal)

#%%
lims = [100,500]
x='Hour'
y='RRP'
height=900
width=1200

diurnal.loc[(diurnal[y] < lims[0]) | (diurnal[y] > lims[1])] = np.nan
fig = px.box(
    diurnal,
    x=x,
    y=y,
    labels={'RRP':'RRP ($/MWh)'},
    title=f"${lims[0]}-${lims[1]}/MWh diurnal energy RRP profile ({region}, FY{int(year)}-{int(year+1)})",
    color_discrete_sequence=plotly.colors.qualitative.Antique,
    log_y=False,
    points=False
    )

fig.update_layout(font={'size':23})
    
saveStr=os.path.join(myPath,resultsDir)
slices = [col for col in [x,y] if col]
filepath = os.path.join(saveStr,f"diurnalPrices_{year}_{region}_{lims[0]}-{lims[1]}_{'_'.join(slices)}")
plot(fig,filename=filepath + '.html')
fig.write_image(filepath + '.png',width=width,height=height)
#%%

def thresh_smooth(rdata,freq,window,thresh,col,roll=False):
    """
    Implements a smoothing function that sets the value of data to the average value across a rolling 
    window if the differential across that rolling window is < thresh.
    
    RRP_6t' = {(1/h) ∑_t^(t+h)_t RRP_6t, |max⁡(RRP_6t)-min(RRP_6t)| < P
            RRP_6t,t∈h,|max⁡(RRP_6t)-min(RRP_6t)| >= P
    Args:
        rdata (pandas Series or pandas DataFrame): Any pandas Series
        
        freq (int): Frequency of the data in mins
        
        window (int): Number of intervals over which to implement rolling average.
        
        thresh (float): Threshold below which to flatten the data.
        
        col (str): Column of your pandas dataframe you want to apply this to, or name you 
            want to give to your series.
        
    Returns:
        threshData (pandas DataFrame): Smoothed data.
        
    Created on 13/11/2020 by Bennett Schneider
    
    """
    rdata = rdata.copy()
    rdata.name = col
    rname = col
    
    try:
        rdata = rdata.to_frame()
    except AttributeError:
        pass
    
    rdata.index = rdata.index - pd.Timedelta(minutes=freq)
    
    if roll:
        rdata = rdata.rolling(window).apply(lambda dataSlice: thresh_flat(dataSlice,thresh,rname,roll=True))
    else:
        rdata['group'] = [i//window for i in range(len(rdata))] # set up a grouping set
        # breakpoint()
        rdata = rdata.groupby('group').apply(lambda dataSlice: thresh_flat(dataSlice,thresh,rname,roll=False))
    rdata = rdata[[rname]]
    rdata.index = rdata.index + pd.Timedelta(minutes=freq)
    
    return rdata
    
    
def thresh_flat(dataSlice,thresh,rname,roll):
    """
    The lambda function for thresh_smooth.
    """
    # print(dataSlice)
    # breakpoint()
    if roll:
        dRange = abs(dataSlice.max() - dataSlice.min())
    else:
        dRange = abs(dataSlice[rname].max() - dataSlice[rname].min())
    # print(dRange)
    # print(dataSlice.mean())
    if roll:
        if dRange < thresh:
            dataSlice = dataSlice.mean()
        else:
            dataSlice = dataSlice[-1]
    else:
        if dRange < thresh:
            dataSlice[rname].loc[:] = dataSlice[rname].mean()
    # print(dataSlice)
    return dataSlice

#%%
smoothData = thresh_smooth(data[['RRP']],30,6,100,'RRP',roll=False)
smoothData.columns = [f"{col}_smooth" for col in smoothData.columns]
#%%
toPlot = pd.concat([data[['RRP']],smoothData[['RRP_smooth']]],axis=1)
WPT.pandasPlotly2(toPlot)