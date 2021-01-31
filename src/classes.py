import os
import sys
import time
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pkl

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
        super(NEM,self).__init__(path)
        self.regionStr='REGIONID'

    def procPrice(self,freq,region,t0,t1,modFunc=None,**kwargs):
        """
        Process pricing data. Takes in rawData from NEM.rawData['Price'] and dumps it into self.procData['Price']
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

        # Add RRP to the procData record
        newData = self.procData['Price'].copy().append(RRP)

        # drop duplicate records before adding to self
        name = newData.index.name
        newData.reset_index(inplace=True)
        newData.drop_duplicates(inplace=True)
        newData.set_index(name,inplace=True)

        self.procData['Price'] = newData

        return RRP

class Gen(object):
    def __init__(self,path,region,Di='/4',Fh='/4',Fr=0.2,Fr_w=None,freq=30,scenario='RRP',modFunc=None,**kwargs):
        """
        Initialises a Gen object.

        Args:
            path (str): Directory you want to save content.

            region (str): Name of the pricing region of the network you want to model.

            Di (float, int, or str. Default='/4'): The dispatch interval. If a str, must be of the form '/{number}'. In this case, Di will be interpreted as the total number of 
                timestamps divided by {number}. Otherwise, it is interpreted as the number of minutes in the dispatch interval.

            Fh (float or int. Default='/4'): The forecast horizon. If a str, must be of the form '/{number}'. In this case, Di will be interpreted as the total number of 
                timestamps divided by {number}. Otherwise, it is interpreted as the number of minutes in the dispatch interval.

            Fr (float. Default=0.2): The regulation dispatch fraction applied to the optimised dispatch. Must be < 1. 

            Fr_w (dict of floats. Default=None): Values are Fr applied to the optimiser when deciding dispatch. Keys are the weights the optimiser 
                asssigns to the respective Fr. If None, equal to {1:Fr}, i.e. the optimiser receives the same setting as is applied to the optimised
                dispatch, with unity weighting.

            freq (float. Default=30): Frequency of the expected input in minutes.

            scenario (str. Default='RRP'): Choose a scenario from the list of scenarios embedded in this function. Must match one of the keys which 
                corresponds to a set of markets across which the object will be optimised.

            modFunc (function. Default=None): Any function whose first input is a pandas dataframe representing a price time-series you want to modify.

            **kwargs (inputs): The remaining inputs for modFunc. If modFunc needs freq, this will be passed without requiring input from kwargs.

        Adds to self as attribute:
            As above

            settings (dict of various): Keys are strings matching the attribute names. Values are the subset of the attributes of self that may be used for
                constructing a unique file name or generating a unique stackable dataframe. 

        Use this function to initialise a Gen class instance.

        Created on 26/01/2021 by Bennett Schneider

        """
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

        self.path = path
        self.region = region
        self.Di = Di
        self.Fh = Fh
        self.Fr = Fr

        # If no weighting set, just apply full weighting on Fr
        if Fr_w:
            self.Fr_w = Fr_w
        else:
            self.Fr_w = {Fr:1}

        self.freq = freq
        self.scenario = scenario
        self.markets = scenarios[scenario]

        self.settings = {
            'region':self.region,
            'Di':self.Di,
            'Fh':self.Fh,
            'Fr':self.Fr,
            'scenario':self.scenario
        }

        self.modFunc = modFunc
        self.kwargs = kwargs

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

        # Construct name based on other settings
        name_settings = '_'.join(list(map(str,self.settings.values())))

        # Combine
        name = f"{dateStr}_{name_settings}"

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

                # iterate through kwargs to get final dataframe
                for key,val in self.kwargs.items():
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
                # Create our input data and store it in a network object
                rrp_mod = Network.procPrice(self.freq,self.region,self.markets,t0,t1,self.scenario,self.modFunc,**self.kwargs)
            else:
                errFunc(e)

        # zero columns based on markets
        for myRRP in [rrp,rrp_mod]:
            if type(myRRP) == pd.core.frame.DataFrame:
                myRRP.drop([col for col in myRRP if 'RRP' not in col],axis=1,inplace=True) # remove metadata
                # Zero by RRP
                for col in myRRP.columns:
                    if col not in self.markets:
                        myRRP[col] = 0

        return rrp,rrp_mod

class BESS(Gen):
    def __init__(self,path,region,Di='/4',Fh='/4',Fr=0.2,freq=30,scenario='RRP',Smax=2,modFunc=None,**kwargs):
        """

        """
        super(BESS,self).__init__(path,region,Di=Di,Fh=Fh,Fr=Fr,freq=freq,scenario=scenario,modFunc=modFunc,**kwargs)
        self.Smax = Smax
        self.settings.update(
            {
                'Smax':Smax
            }
        )

    def optDispatch(self,Network,m,t0,t1,debug=False):
        """
        Uses the metadata stored in the caller to set the inputs for horizonDispatch, which optimises the BESS based on
        RRP, where RRP contains both the original and

        Args:
            t0 (datetime): Starting datetime.

            t1 (datetime): Ending datetime. 
        """

        # Pull data from Network such that it can be directly optimised based on the caller's preset settings
        rrp,rrp_mod = self.getRRP(Network,t0,t1)

        # Get the dispatch interval
        if type(self.Di) == str: # if a string, should be of the form '/num'
            Di = int(round(len(rrp)/float(self.Di.replace('/','')),0)) # interpret as fraction of total number of intervals
        else:
            Di = self.Di # otherwise, interpret as number of minutes
        
        # Get the forecast horizon
        if type(self.Fh) == str: # if a string, should be of the form '/num'
            Fh = int(round(len(rrp)/float(self.Fh.replace('/','')),0)) # interpret as fraction of total number of intervals
        else:
            Fh = self.Fh # otherwise, interpret as number of hours

        breakpoint()
        results = horizonDispatch(rrp,m,self.freq,Fh,Di,sMax=self.Smax,st0=self.Smax/2,eta=1,rMax=1,regDisFrac=self.Fr,regDict=self.Fr_w,debug=debug,rrp_mod=rrp_mod)

        # Return the RRP signal passed through the optimiser as well as the one that is evaluated. Return as a stacked dataframe where
        # the original is flagged with the word 'Orig' and the one from the optimiser is flagged according to the algorithm used.

        return results