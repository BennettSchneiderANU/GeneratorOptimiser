import os
import sys
import time
from numpy.lib.function_base import disp
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pkl
from ipywidgets import widgets
import copy

from pandas.core.frame import DataFrame
# import opennem

from analysis_functions import *
from error_handling import *

class Network(object):
    def __init__(self,path):
        """
        Initialise network object
        """
        self.rawData = {}
        self.procData = {}
        self.Gens = {}
        self.rawFreq = {}
        self.path = path
        
    
    def loadRaw(self,rawData,freq,name):
        """
        Load in network data
        """

        self.rawData[name] = rawData.copy()
        self.procData[name] = pd.DataFrame() # create empty matching procData entry 
        self.rawFreq[name] = freq

class NEM(Network):
    """
    NEM-specific version of Network object.
    """

    def __init__(self,path):
        super().__init__(path)
        self.regionStr='REGIONID'

    # def loadOpenNEM(self,start,end,region=None):
    #     """
    #     Loads data from opennem.
    #     """
    #     if region:
    #         data_5min = opennem.web_api.load_data(d1=start,d2=end,region=self.region.lower(),split=False).fillna(method='bfill')

    def procPrice(self,freq,region,t0,t1,pivot=False,modFunc=None,**kwargs):
        """
        Process pricing data. Takes in rawData from NEM.rawData['Price'], averages to a given frequency, modifies with modFunc,
        converts to stacked schema, and dumps it into self.procData['Price'].

        Args:
            freq (int): Frequency of the output data in minutes.

            region (str): Filter on self.rawData['Price']['REGIONID']

            t0 (datetime): The start of the desired slice of the raw data.

            t1 (datetime): THe end of the desired slice of the raw data.

            pivot (bool. Default=False): If True, outputs the pivoted price according to the given market to which it applies. If false, outputs as a stacked
                table consisting of 'Market' and 'RRP' columns. This modification DOES NOT apply to the version of the data which is stored in the caller under self.procData['Price'],
                which is always fully stacked.

            modFunc (function): Function used to modify the raw data. Flagged as modified according to the function name in addition to the function inputs

            **kwargs (dict of inputs): Inputs for modFunc. Freq is often also an input for modFunc, but it needn't be included here, as it is included based on the freq input to NEM.procPrice()

        Retuns:
            RRP (pandas DataFrame): self.rawData['Price'] averaged to freq, filtered by region and t0/t1. If modFunc is defined, also outputs the same output as modified by modFunc, stacked in the
                dataframe and flagged by the name of modFunc and by its kwargs. If pivot = True, RRP is split out by market in the columns. Otherwise, output as fully stacked dataframe under two 
                separate columns, RRP and Market.
        
        Adds to self as attribute:
            self.procData['Price'] (pandas DataFrame): Adds RRP in fully stacked form to the already existing self.procData['Price']. All duplicates are removed.

        Use this function to generate processed RRP data from the raw record.

        Created on 14/02/2021 by Bennett Schneider 

        """

        try:
            RRP = self.rawData['Price'].copy()
        except KeyError:
            errFunc("NEM.rawData['Price'] doesn't exist! You need to load raw price data into your NEM instance using NEM.load().")

        # Filter by region
        RRP = RRP[RRP['REGIONID'] == region]

        # # Zero by RRP
        # for col in RRP.columns:
        #     if col not in markets:
        #         RRP[col] = 0

        # Filter by time
        RRP = RRP.loc[(RRP.index >= t0) & (RRP.index <= t1)]

        # Average to self.freq
        RRP = timeAvgEnd(RRP[[col for col in RRP if col != 'REGIONID']],self.rawFreq['Price'],freq)
        
        if modFunc:
            # most functions will require freq as well, which is already an input, but do
            # not force this to be the case
            try:
                kwargs.update({'freq':freq})
                RRP_mod = modFunc(RRP,**kwargs)
            except TypeError:
                RRP_mod = modFunc(RRP,**kwargs)
                
            # further mark RRP_mod by the kwargs inputs
            for key,val in kwargs.items():
                RRP_mod[key] = val

        # mark RRP as the original
        RRP['modFunc'] = 'Orig'
        
        if modFunc: # concatenate the two dfs together
            RRP = pd.concat([RRP,RRP_mod],sort=True)
        
        # add common markers
        RRP['REGIONID'] = region

        # RRP['scenario'] = scenario

        RRP['freq'] = freq

        # stack
        name = RRP.index.name
        RRP = RRP.reset_index()
        RRP.set_index([col for col in RRP if 'RRP' not in col],inplace=True)
        RRP.columns = [col.replace('RRP','') for col in RRP] # drop RRP from index so we can have a common 'Market' key
        RRP.rename({'':'Energy'},axis=1,inplace=True) # convert what was originally RRP to 'Energy'
        RRP = RRP.stack().reset_index().rename({f'level_{len(RRP.index.names)}':'Market',0:'RRP'},axis=1).set_index(name) # stack by market as well

        # Add RRP to the procData record
        newData = self.procData['Price'].copy().append(RRP)

        # drop duplicate records
        newData.reset_index(inplace=True)
        newData.drop_duplicates(inplace=True)
        newData.set_index(name,inplace=True)

        # add to self
        self.procData['Price'] = newData

        # Unstack the output (stacked output only for self.procData)
        if pivot:
            RRP.reset_index(inplace=True)
            RRP = pd.pivot_table(RRP.fillna(999),values='RRP',index=[col for col in RRP if col not in ['Market','RRP']],columns='Market').reset_index().set_index('Timestamp').rename_axis(None,axis=1)

        return RRP

class Gen(object):
    def __init__(self,path,region,name,t0,t1,version=''): # Di='/4',Fh='/4',Fr=0.2,Fr_w=None,freq=30,scenario='RRP',modFunc=None,**kwargs):
        """
        Initialises a Gen object.

        Args:
            path (str): Directory you want to save content.

            region (str): Name of the pricing region of the network you want to model.

            name (str): Name of the generator. Must match a column in configs/inputs.csv.

            version (str. Default=''): If you have a different version of the input file, it must be placed in the configs folder in the repo and named
                inputs_[version].csv

        Adds to self as attribute:
            As above

            config (pandas DataFrame): See self.loadConfig()
            
            Each of the variables in the 'Input' column of inputs.csv will be set as an attribute of self. See self.loadConfig()

            markets (list of strs): Based on the value of scenario. 

            results (pandas DataFrame): Empty dataframe for storing dispatch results.

            results (pandas DataFrame): Empty dataframe for storing operational dispatch results.

        Use this function to initialise a Gen class instance.

        Created on 26/01/2021 by Bennett Schneider

        """
        scenarios = {
            'Energy': ['Energy'],
            'FCASCoopt': ['Energy','RAISE6SEC','RAISE60SEC','RAISE5MIN','RAISEREG','LOWER6SEC','LOWER60SEC','LOWER5MIN','LOWERREG'],
            'RegCoopt': ['Energy','RAISEREG','LOWERREG'],
            'RegRaiseCoopt': ['Energy','RAISEREG'],
            'RegLowerCoopt': ['Energy','LOWERREG'],
            'ContCoopt': ['Energy','RAISE6SEC','RAISE60SEC','RAISE5MIN','LOWER6SEC','LOWER60SEC','LOWER5MIN'],
            'ContRaiseCoopt': ['Energy','RAISE6SEC','RAISE60SEC','RAISE5MIN'],
            'ContLowerCoopt': ['Energy','LOWER6SEC','LOWER60SEC','LOWER5MIN']
            }

        self.path = path
        self.region = region
        self.name = name
        self.t0 = t0
        self.t1 = t1
        self.version = version

        self.loadConfig(version=version)

        # If no weighting set, just apply full weighting on Fr
        if self.RFr:
            self.RFr = self.RFr
        else:
            self.RFr = {self.RFr:1}

        if self.LFr:
            self.LFr = self.LFr
        else:
            self.LFr = {self.LFr:1}

        # Lookup markets
        self.markets = scenarios[self.scenario]
        
        # For storing dispatch results
        self.revenue = pd.DataFrame()
        self.operations = pd.DataFrame()
        
        # ModFunc should be reset to None if nan and to a function otherwise
        try:
            if np.isnan(self.modFunc):
                self.modFunc = None
        except TypeError:
            self.modFunc = eval(self.modFunc)

        # Convert to function object
        if self.optfunc:
            self.optfunc = eval(self.optfunc)

    def loadConfig(self,version=''):
        """
        Reads the master input config containing all settings. Sets the variable in each line as an attribute of self. Uses the value entered under self.name,
        but will fill values not entered with the default settings.

        Args:
            self (Generator):
              - name (str): Name of the generator.
            
            version (str. Default=''): If you have a different version of the input file, it must be placed in the configs folder in the repo and named
                inputs_[version].csv

        Adds to self as attribute:
            config (pandas DataFrame): The following columns of input.csv: 'Input','Object','Default',self.name

            Input (attributes): Each of the variables in the 'Input' column of inputs.csv will be set as an attribute of self.

            marketmeta (pandas DataFrame): Market metdata, principally for constructing robust schema for results.

        Use this function to conveniently define a generator object based on a myriad of inputs. This function is called in Generator.__init__().

        Created on 26/06/2021 by Bennett Schneider

        """
        config_path = os.path.join(os.path.dirname(os.path.dirname((__file__))),'configs','inputs' + version + '.csv') # construct path
        config = pd.read_csv(config_path,index_col=0)[['Object','Default',self.name]] # read in the config data as a pandas df
        config['values'] = config[self.name].fillna(config['Default']) # if nan, fill with default values
        self.config = config # assign to self
        
        # Set these inputs as attributes
        for i,val in self.config['values'].to_dict().items():
            try:
                val = float(val)
            except ValueError:
                pass
            setattr(self,i,val)   

        # read in the market metadata lookup as well
        marketmeta_path = os.path.join(os.path.dirname(os.path.dirname((__file__))),'configs','marketmeta.csv') # construct path
        marketmeta = pd.read_csv(marketmeta_path) # read in the markets data as a pandas df
        self.marketmeta = marketmeta

    def pickle(self,nowStr="%Y%m%d",selfStr="%Y%m%d",comment=None):
        """
        Pickles Gen object to path, using today's date to differentiate it.

        Args:
            self (Gen): 
                - settings (dict): Set of settings that can uniquely define the caller.
                - t0 (datetime): Start time.
                - t1 (datetime): End time.
                - path (str): Directory you want to save data in.

                nowStr (str. Default="%Y%m%d"): String format of today's date.

                selfStr (str Default="%Y%m%d"): String format of t0 and t1, the start and end dates.
                
                comment (str. Default=None): An additional field which allows you to add an additional comment

        Returns:
            pickle_path (str): Location of the pickle file saved.
        
        Pickles caller to pickle_path.

        Created on 26/01/2021 by Bennett Schneider

        """
        # create the pickle path based on the fileName
        name = self.fileName(nowStr=nowStr,selfStr=selfStr,comment=comment)
        pickle_path = os.path.join(self.path,f"{name}.pkl")
        fileObj = open(pickle_path,'wb') # create file object for pickling
        print('\n**************************************')
        print(f'Pickling object to:\n{pickle_path}')
        print('**************************************\n')

        pkl.dump(self,fileObj,protocol=0) # pickle self

        fileObj.close()

        return pickle_path

        
    def fileName(self,nowStr="%Y%m%d",selfStr="%Y%m%d",comment=None):
        """
        Creates fileName based on settings and timeStr of current time.

        Args:
            self (Gen):
              - settings (dict): Set of settings that can uniquely define the caller.
              - t0 (datetime): Start time.
              - t1 (datetime): End time.
            
            nowStr (str. Default="%Y%m%d"): String format of today's date.

            selfStr (str Default="%Y%m%d"): String format of t0 and t1, the start and end dates.

            comment (str. Default=None): An additional field which allows you to add an additional comment

        Returns:
            name (str): Filename, not including the full path.

        Created on 26/01/2021 by Bennett Schneider
        
        """
        # get today's date
        dateStr = exportTimestamp(timeStr=nowStr)

        # Get dates
        start = self.t0.strftime(selfStr)
        end = self.t1.strftime(selfStr)

        # Combine
        name = f"{dateStr}_{self.name}_{start}-{end}"

        # Add comment
        if comment:
            name += f"_{comment}"

        return name

    def getRRP(self,Network,t0,t1):
        """
        Uses Gen settings to extract the desired actual price and modified price from the Network object.

        Args:
            self (Gen):
              - region
              - scenario
              - freq
              - modFunc
              - kwargs
              - markets

            Network (Network): 
              - regionStr
              - procData['Price']
        
        Functions used:
            classes:
              - Network.procPrice()
        
        Returns:
            rrp (pandas DataFrame): The actual rrp associated with the caller's frequency, scenario (markets) and region.

            rrp_mod (pandas DataFrame): The modified rrp associated with the actual rrp as well as modFunc and its associated kwargs.

        Use this function to extract the relevant rrp data for a Gen object from a Network object.

        Created on 31/01/2021 by Bennett Schneider
        """


        try:
            nPrice = Network.procData['Price'].copy()

            # Pivot the stacked table for convenience going forward
            nPrice.reset_index(inplace=True)

            nPrice = pd.pivot_table(nPrice.fillna(999),values='RRP',index=[col for col in nPrice if col not in ['Market','RRP']],columns='Market').reset_index().set_index('Timestamp').rename_axis(None,axis=1)

            # logic required to slice out the correct data from the network price record
            logic = (nPrice[Network.regionStr] == self.region) & \
                    (nPrice['freq'] == self.freq) & \
                    (nPrice.index > t0) & \
                    (nPrice.index <= t1)

            # This is how long the result should be if all the timestamps are present
            expLen = int((t1 - t0).total_seconds()/60/self.freq)
            
            if self.modFunc:
                # Additional logic required to pull the modified price record
                mod_logic = nPrice['modFunc'] == self.modFunc.__name__

                # use the logic to slice the data
                rrp_mod = nPrice[logic & mod_logic]

                # Get the kwargs for optfunc to be passed through horizonDispatch
                kwargs = self.extractKwargs(Network.procPrice,self.modFunc)

                # iterate through kwargs to get final dataframe
                for key,val in kwargs.items():
                    rrp_mod = rrp_mod[rrp_mod[key] == val] 

                # Raise an error if we don't have the right amount of data
                if len(rrp_mod) != expLen:
                    raise IndexError
            else:
                rrp_mod = None
            
            orig_logic = nPrice['modFunc'] == 'Orig'
            # use the logic to slice the data
            rrp = nPrice[logic & orig_logic]

            # Raise an error if we don't have the right amount of data
            if len(rrp) != expLen:
                raise IndexError

        except IndexError as e:
            if self.modFunc:
                # Get the kwargs for optfunc to be passed through horizonDispatch
                kwargs = self.extractKwargs(Network.procPrice,self.modFunc)
                # Create our input data and store it in a network object
                RRP = Network.procPrice(self.freq,self.region,t0,t1,True,self.modFunc,**kwargs)
                rrp = RRP[RRP['modFunc'] == 'Orig']
                rrp_mod = RRP[RRP['modFunc'] != 'Orig']
            else:
                errFunc(e)
        except (KeyError,TypeError) as e:
            if self.modFunc:
                kwargs = self.extractKwargs(Network.procPrice,self.modFunc)
            else:
                kwargs = {}
            RRP = Network.procPrice(self.freq,self.region,t0,t1,True,self.modFunc,**kwargs)
            rrp = RRP[RRP['modFunc'] == 'Orig']
            rrp_mod = RRP[RRP['modFunc'] != 'Orig']
        except TypeError:
            errFunc("Reload your Network object!")
    
        # zero columns based on markets
        my_rrp = [rrp,rrp_mod]
        my_new_rrp = []
        # List of markets listed in the processed data
        market_list = list(Network.procData['Price']['Market'].unique())
        for myRRP in my_rrp:
            if type(myRRP) == pd.core.frame.DataFrame:
                rrp_copy = myRRP.copy()
                rrp_copy.drop([col for col in rrp_copy if col not in market_list],axis=1,inplace=True) # remove metadata
                # Zero by RRP
                for col in rrp_copy.columns:
                    if col not in self.markets:
                        rrp_copy.loc[:,col] = 0
            else:
                rrp_copy = myRRP
                
            my_new_rrp.append(rrp_copy) # append zeroed data to new rrp list
                
        rrp,rrp_mod = my_new_rrp # assign back to original dataframes

        return rrp,rrp_mod

    def extractKwargs(self,caller,called):
        """
        Extracts the kwargs inputs from self such that it can be passed to the called function as a **kwargs argument through the caller
        without duplicating inputs, e.g. caller(input1, input2,...,inputi,**kwargs) -> called(inputi,**kwargs).
        Attributes of self must match the names of the inputs such that they can be extracted with getattr(self,"inputi") = self.inputi.

        Args:
            self (Generator):
              - name (str): Name of the generator.
              - All names if the inputs of called that are not already arguments of caller.

            caller (func): Any function.

            called (func): Any function.

        Returns:
            kwargs (dict of inputs): Keys are the names of the inputs as strings. Values are the values of those inputs.

        Use this function to construct a valid kwargs dictionary for a given function called inside another function, which may share inputs.

        Created on 28/06/2021 by Bennett Schneider
            
        """

        caller_inputs = inspect.getfullargspec(caller).args
        called_inputs = inspect.getfullargspec(called).args

        # pull kwargs out of the attributes of self that match the optional arguments of optfunc if they are not already used in horizon dispatch
        kwargs = {}
        for i in called_inputs:
            if i not in caller_inputs:
                try:
                    kwargs[i] = getattr(self,i)
                except AttributeError:
                    logging.debug(f" {i} is missing from {self.name} object. Using default value")

        return kwargs

class BESS(Gen):
    def __init__(self,path,region,name,t0,t1,**kwargs):
        """
        Same as parent __init__().
        """
        
        super().__init__(path,region,name,t0,t1,**kwargs)

    def optDispatch(self,Network,m): # ,optfunc=BESS_COINOR,**kwargs):
        """
        Uses the metadata stored in the caller to set the inputs for horizonDispatch and the bess optimiser (optfunc). This will optimise battery dispatch
        based on RRP. If self.modFunc points to a suitable function, the resulting modified RRP will be stored in Network and used to feed the objective
        function of the optimiser. The actual RRP will then be used to evaluate the result.

        Args:
            self (BESS): 
              - freq (int)
              - Di (str or int)
              - Fh (str or int)
              - optfunc (func)
              - sMax (float)
              - any additional inputs for optfunc
              - t0 (datetime): Starting datetime.
              - t1 (datetime): Ending datetime. 

            Network (Network): See classes.Network.__init__()

        Functions used:
            classes:
              - Generator.extractKwargs()
              - Generator.getRRP()
            
            analysis_functions:
              - horizonDispatch()
            
            other:
              - optfunc()
              - modFunc()
        
        Adds to self as attribute:
            revenue (pandas DataFrame): The revenue earned in each market due to optimal battery dispatch. Appends to existing if already exists. 

            operations (pandas DataFrame): The operations the battery took in each market to achieve optimal dispatch. Appends to existing if already exists.

        Use this functions to optimally dispatch a battery using a given dispatch function.

        Created on 29/06/2021 by Bennett Schneider

        """

        # Pull data from Network such that it can be directly optimised based on the caller's preset settings
        rrp,rrp_mod = self.getRRP(Network,self.t0,self.t1)

        # Get the dispatch interval
        if type(self.Di) == str: # if a string, should be of the form '/num'
            Di = int(round(len(rrp)/float(self.Di.replace('/','')),0)) # interpret as fraction of total number of intervals
            Di *= self.freq # convert from number of intervals to minutes
        else:
            Di = self.Di # otherwise, interpret as number of minutes
        
        # Get the forecast horizon
        if type(self.Fh) == str: # if a string, should be of the form '/num'
            Fh = int(round(len(rrp)/float(self.Fh.replace('/','')),0)) # interpret as fraction of total number of intervals
            Fh *= self.freq/60 # convert from number of intervals to hours
        else:
            Fh = self.Fh # otherwise, interpret as number of hours
        
        # Get the kwargs for optfunc to be passed through horizonDispatch
        kwargs = self.extractKwargs(horizonDispatch,self.optfunc)

        revenue,operations = horizonDispatch(rrp,m,self.freq,Fh,Di,optfunc=self.optfunc,st0=self.sMax/2,rMax=1,rrp_mod=rrp_mod,**kwargs)

        if len(self.revenue) == 0:
            self.revenue = revenue.copy()
        else:
            self.revenue = self.revenue[~(self.revenue.index > self.t0) | ~(self.revenue.index <= self.t1)] # remove any existing data from current time range
            self.revenue = self.revenue.append(revenue).sort_index() # append new calculations
        
        if len(self.operations) == 0:
            self.operations = operations.copy()
        else:
            self.operations = self.operations[~(self.operations.index > self.t0) | ~(self.operations.index <= self.t1)] # remove any existing data from current time range
            self.operations = self.operations.append(operations).sort_index() # append new calculations

    def stackRevenue(self,setIndex=False,load=True):
        """
        Fully stacks revenue attribute for plotting convenience.
        """
        name = self.revenue.index.name
        stackedRevenue = self.revenue.copy().reset_index().set_index(['Timestamp','Market']).stack().reset_index().rename({'level_2':'Variable',0:'Value'},axis=1)
        if setIndex:
            stackedRevenue.set_index(name,inplace=True)
        
        if load:
            self.stackedRevenue = stackedRevenue.copy()
        return stackedRevenue

    def stackOperations(self,setIndex=False,load=True):
        """
        Fully stacks operations attribute for plotting convenience.
        """
        name = self.operations.index.name
        stackedOperations = self.operations.copy().stack().reset_index().rename({'level_1':'Variable',0:'Value'},axis=1)
        if setIndex:
            stackedOperations.set_index(name,inplace=True)
        
        if load:
            self.stackedOperations = stackedOperations.copy()
        return stackedOperations

    def rrpSchema(self,Network,t0=None,t1=None):
        """
        Gets the price signal from the network object and maps on market-specific metadata. Then stacks the data.

        Args:
            self (Generator):
              - marketmeta

            Network (Network)

            t0 (datetime. Default=None): Start timestamp (open bound)
            
            t1 (datetime. Default=None): End timestamp (closed bound)

        Functions used:
            classes:
              - Generator.getRRP()

        Returns:
            RRP_schema (pandas DataFrame): Stacked version of price dataframe from self.getRRP() with metadata from marketmeta merged on.

        Use this function for pre-processing for plotting.

        Created on 3/07/2021 by Bennett Schneider 
        """
        # Get the price
        rrp,rrp_mod = self.getRRP(Network,t0,t1)
        rrp['Version'] = 'Orig'
        # If rrp_mod, combine the two
        if type(rrp_mod) == pd.core.frame.DataFrame:
            rrp_mod['Version'] = 'Mod'
            RRP = pd.concat([rrp,rrp_mod])
        else:
            RRP = rrp

        RRP_scenario = pd.DataFrame()
        # Collect the price scenarios we are interested in
        for scenario in self.marketmeta['Scenario'].unique():
            RRP_scenario[scenario] = RRP[list(self.marketmeta.loc[self.marketmeta['Scenario'] == scenario,'Market'])].sum(axis=1)
            
        # Remove the prices not in the given scenario
        RRP_scenario = RRP_scenario[[col for col in RRP_scenario.columns if RRP_scenario[col].abs().sum() > 0]]
        
        # Get the version
        RRP_scenario['Version'] = RRP['Version']
        
        # Stack RRP
        RRP_scenario = RRP_scenario.reset_index().set_index(['Timestamp','Version']).stack().reset_index().rename({'level_2':'Scenario',0:'RRP'},axis=1)

        # Apply metadata
        RRP_schema = pd.merge(RRP_scenario,self.marketmeta,how='inner',on='Scenario')
        
        return RRP_schema

    

    # def plotOutput(self,Network,t0,t1):
    #     """
    #     """
    #     rrp,rrp_mod = self.getRRP(Network,t0,t1)

        

    #     self.stackRevenue()
    #     self.stackOperations()

    # def energyRevenuePlot(self,Network,show=True,t0=None,t1=None,kwargs={}):
    #     """
    #     """

    #     # Get the revenue table
    #     revenue = self.revenue.copy()

    #     # Filter for energy and pull out the desired column
    #     revenue = revenue[revenue['Market'] == 'Energy']['Revenue_$'].to_frame('Revenue')

    #     if t0 and t1:
    #         revenue = revenue[(revenue.index > t0) & (revenue.index <= t1)]

    #     else:
    #         # Get the price data for the given interval
    #         t0 = revenue.index.min()
    #         t1 = revenue.index.max()

    #     rrp,rrp_mod = self.getRRP(Network,t0,t1)
        
    #     # If rrp_mod, combine the two
    #     if type(rrp_mod) == pd.core.frame.DataFrame:
    #         rrp = rrp['Energy'].to_frame('Energy Price (Actual)')
    #         rrp_mod = rrp_mod['Energy'].to_frame('Energy Price (Objective)')
    #         RRP = pd.concat([rrp,rrp_mod],axis=1)
    #     else:
    #         RRP = rrp['Energy'].to_frame('Energy Price')

    #     # Combine
    #     toPlot = pd.concat([RRP,revenue],axis=1)

    #     fig = plotlyPivot(
    #         toPlot,
    #         Scatter=['Revenue'],
    #         Line=list(RRP.columns),
    #         colorList = plotly.colors.qualitative.Pastel,
    #         fill={'Revenue':'tozeroy'},
    #         secondary_y=['Revenue'],
    #         show=show,
    #         opacity=0.4,
    #         **kwargs,
    #         )

    #     return fig


    def plotDispatch(self,Network,show=True,t0=None,t1=None,**kwargs):
        """
        Stacked bar of RegDt_MW and dt_MW
        Filled scatter of storage level (MWh)
        """

        # Get the operations table
        operations = self.operations.copy()
        # operations = operations.rename({'dt_MW':'Energy dispatch','regDt_MW': 'Reg Energy Dispatch','st_MWh': 'State of Charge'},axis=1)
        operations = operations[[col for col in operations.columns if operations[col].abs().sum() > 1]] # filter out empty columns

        # Get the revenue table
        revenue = self.revenue.copy()

        if t0 and t1:
            operations = operations[(operations.index > t0) & (operations.index <= t1)]
            revenue = revenue[(revenue.index > t0) & (revenue.index <= t1)]
        else:
            t0 = self.t0
            t1 = self.t1

        # Process revenue table
        dispatch = revenue[['Market','Dispatch_MW']] # Only keep dispatch
        dispatch = pd.merge(dispatch.reset_index(),self.marketmeta,how='inner',on='Market') # map on metadata
        dispatch.loc[dispatch['Direction'] == 'LOWER','Dispatch_MW'] = -dispatch['Dispatch_MW'] # set lower markets to negative

        dispatch = dispatch.pivot_table(index='Timestamp',columns='Scenario',values='Dispatch_MW',aggfunc='mean') # pivot by scenario
        dispatch.drop('Energy',axis=1,inplace=True) # remove energy, as this is made up of dt_MW and regDt_MW
        dispatch.columns = [f"Dispatch_{col}" for col in dispatch.columns] # rename the columns
        dispatch = dispatch[[col for col in dispatch if dispatch[col].abs().sum() > 1]] # remove the columns which weren't activated

        bars = [col for col in operations if 'st_MWh' not in col] # all the columns of operations save st_MWh
        bars.extend(list(dispatch.columns))

        # Get the rrp
        rrp,rrp_mod = self.getRRP(Network,t0,t1)
        rrp = rrp[[col for col in rrp if rrp[col].sum() > 0]] # remove empty coluns
        lines = list(rrp.columns)
        
        if rrp_mod:
            rrp_mod = rrp_mod[[col for col in rrp_mod if rrp_mod[col].sum() > 0]] # remove empty columns
            rrp_mod.columns = [f"{col}_mod" for col in rrp_mod.columns] # rename mods
            lines.extend(list(rrp_mod.columns))
        
        toPlot = pd.concat([operations,dispatch,rrp,rrp_mod],axis=1)

        # pastel, but insert black in 2nd position to serve as regDt, so line and bar markets align
        line_colors = copy.deepcopy(list(plotly.colors.qualitative.Pastel))
        scatter_colors = ['rgb(255,0,0)']
        bar_colors = copy.deepcopy(list(plotly.colors.qualitative.Pastel))
        if 'regDt_MW' in toPlot.columns:
            bar_colors.insert(1,'#222A2A')

        fig = plotlyPivot(
            toPlot,
            Scatter=['st_MWh'],
            Line=lines,
            Bar=bars, 
            relative=True,
            colorList = {
                'Bar': bar_colors,
                'Scatter': scatter_colors,
                'Line': line_colors
            },
            # fill={'st_MWh':'tozeroy'},
            secondary_y=lines,
            show=show,
            opacity=1,
            scatterLineWidth=2,
            title=self.name,
            **kwargs
            )
        
        return fig

    # def rrpPlot(self,Network,show=True,t0=None,t1=None):
    #     """
    #     """
    #     # Process the rrp data into the desired schema
    #     RRP_plot = self.rrpSchema(Network,t0=t0,t1=t1)
        
    #     # Construct a plot with plotly express based on just the price
    #     fig = px.line(RRP_plot,x='Timestamp',y='RRP',color='Direction',line_dash='Version',facet_row='Category').update_yaxes(matches=None)
    #     if show:
    #         plot(fig)

    # def cooptDispatchPlot(self,show=True,t0=None,t1=None):
    #     """
    #     """

    #     # Get the revenue table
    #     revenue = self.revenue.copy()
        
    #     if t0 and t1:
    #         revenue = revenue[(revenue.index > t0) & (revenue.index <= t1)]
        
    #     revenue = pd.merge(revenue.reset_index(),self.marketmeta,how='inner',on='Market')
    #     revenue.loc[revenue['Direction'] == 'LOWER','Dispatch_MW'] = -revenue['Dispatch_MW']
    
    #     fig = px.bar(revenue,x='Timestamp',y='Dispatch_MW',color='Direction',facet_row='Category',barmode='relative').update_yaxes(matches=None)
        
    #     if show:
    #         plot(fig)
        
    # def cooptRevenuePlot(self,show=True,t0=None,t1=None):
    #     """
    #     """

    #     # Get the revenue table
    #     revenue = self.revenue.copy()
        
    #     if t0 and t1:
    #         revenue = revenue[(revenue.index > t0) & (revenue.index <= t1)]
        
    #     revenue = pd.merge(revenue.reset_index(),self.marketmeta,how='inner',on='Market')
    
    #     fig = px.bar(revenue,x='Timestamp',y='Revenue_$',color='Direction',facet_row='Category',barmode='relative').update_yaxes(matches=None)
        
    #     if show:
    #         plot(fig)

    # def plotEnergy(self,Network,t0=None,t1=None,dipatch_kwargs={},revenue_kwargs={}):
    #     """
    #     Needs some work to preserve the formatting of revenue and dispatch figures
    #     """

    #     # Get the operations table
    #     operations = self.operations.copy()

    #     # Get the revenue table
    #     revenue = self.revenue.copy()

    #     # Filter for energy and pull out the desired column
    #     revenue = revenue[revenue['Market'] == 'Energy']['Revenue_$'].to_frame('Revenue')


        # fig = make_subplots(rows=2, cols=1,shared_xaxes=True)

        # for d in dispatch.data:
        #     fig.add_trace(
        #         d,
        #         row=1, col=1
        #     )

        # for d in revenue.data:
        #     fig.add_trace(
        #         d,
        #         row=2, col=1
            # )

        return fig

    def regRevenuePlot():
        """
        """
    
    def regDispatchPlot():
        """
        """
    

    def contRevenuePlot():
        """
        """
    
    def contDispatchPlot():
        """
        """


