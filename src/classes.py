import os
import sys
import time
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pkl
from mip import *

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

    def procPrice(self,freq,region,markets,t0,t1,scenario,modFunc,**kwargs):
        """
        Process pricing data. Takes in rawData from NEM.rawData['Price'] and dumps it into self.procData['Price']
        """
        try:
            RRP = self.rawData['Price'].copy()
        except KeyError:
            errFunc("NEM.rawData['Price'] doesn't exist! You need to load raw price data into your NEM instance using NEM.load().")

        # Filter by region
        RRP = RRP[RRP['REGIONID'] == region]

        # Zero by RRP
        for col in RRP.columns:
            if col not in markets:
                RRP[col] = 0

        # Filter by time
        RRP = RRP.loc[(RRP.index >= t0) & (RRP.index <= t1)]

        # Average to self.freq
        RRP = timeAvgEnd(RRP[[col for col in RRP if col != 'REGIONID']],self.rawFreq['Price'],freq)

        RRP_mod = modFunc(RRP,**kwargs)

        RRP['modFunc'] = 'Orig'

        RRP = pd.concat([RRP,RRP_mod])

        RRP['REGIONID'] = region

        RRP['Scenario'] = scenario

        RRP['freq'] = freq

        # Add RRP to the procData record
        self.procData['Price'].append(RRP)

        return RRP

class Gen(object):
    def __init__(self,path,region,Di='/4',Fh='/4',Fr=0.2,Fr_w=None,freq=30,scenario='RRP'):
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
            self.Fr_w = {1:Fr}

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

class BESS(Gen):
    def __init__(self,path,t0,t1,region,Di=5,Fh=0.25,Fr=0.2,freq=30,scenario='RRP',Smax=2):
        """

        """
        super(BESS,self).__init__(path,t0,t1,region,Di=Di,Fh=Fh,Fr=Fr,freq=freq,scenario=scenario)
        self.Smax = Smax
        self.settings.update(
            {
                'Smax':Smax
            }
        )

    def optDispatch(self,Network,m,t0,t1,debug=False,modFunc=None,**kwargs):
        """
        Uses the metadata stored in the caller to set the inputs for horizonDispatch, which optimises the BESS based on
        RRP, where RRP contains both the original and

        Args:
            t0 (datetime): Starting datetime.

            t1 (datetime): Ending datetime. 
        """

        try:
            nPrice = Network.procData['Price'].copy()

            # logic required to slice out the correct data from the network price record
            logic = (nPrice['REGIONID'] == self.region) & \
                    (nPrice['scenario'] == self.scenario) & \
                    (nPrice['freq'] == self.freq) & \
                    (nPrice.index > t0) & \
                    (nPrice.index <= t1)

            # This is how long the result should be if all the timestamps are present
            expLen = int((t1 - t0).total_seconds()/60/self.freq)
            
            if modFunc:
                # Additional logic required to pull the modified pric record
                mod_logic = nPrice['modFunc'] == modFunc.__name__

                # use the logic to slice the data
                rrp_actual = nPrice[logic & mod_logic]

                # Raise an error if we don't have the right amount of data
                if len(rrp_actual) != expLen:
                    raise IndexError
            
            orig_logic = nPrice['modFunc'] == 'Orig'
            # use the logic to slice the data
            rrp = nPrice[logic & orig_logic]

            # Raise an error if we don't have the right amount of data
            if len(rrp) != expLen:
                raise IndexError

        except IndexError:      
            # Create our input data and store it in a network object
            rrp_mod = Network.procPrice(self.freq,self.region,self.markets,t0,t1,self.scenario,modFunc,**kwargs)

        rrp = rrp_mod[rrp_mod['modFunc'] == 'Orig'] # slice of non-modded rrp

        # If modFunc was passed, slice that out as well and assign as rrp_actual, otherwise set to None
        if modFunc:
            rrp_actual = rrp_mod[rrp_mod['modFunc'] == modFunc.__name__] # slice of modded rrp
        else:
            rrp_actual = None

        # Get the dispatch interval
        if type(self.Di) == str: # if a string, should be of the form '/num'
            Di = len(rrp)/float(self.Di.replace('/','')) # interpret as fraction of total number of intervals
        else:
            Di = self.Di # otherwise, interpret as number of minutes
        
        # Get the forecast horizon
        if type(self.Fh) == str: # if a string, should be of the form '/num'
            Fh = len(rrp)/float(self.Fh.replace('/','')) # interpret as fraction of total number of intervals
        else:
            Fh = self.Fh # otherwise, interpret as number of hours


        results = horizonDispatch(rrp,m,self.freq,Fh,Di,sMax=self.Smax,st0=self.Smax/2,eta=1,rMax=1,regDisFrac=self.Fr,regDict=self.Fr_w,debug=debug,rrp_actual=rrp_actual)

        # Return the RRP signal passed through the optimiser as well as the one that is evaluated. Return as a stacked dataframe where
        # the original is flagged with the word 'Orig' and the one from the optimiser is flagged according to the algorithm used.

        return results