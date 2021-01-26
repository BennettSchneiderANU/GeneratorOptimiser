import os
import sys
import time
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pkl

from analysis_functions import *
from error_handling import *

class Gen(object):
    def __init__(self,path,t0,t1,region,Di=0.25,Fh=0.25,Fr=0.2,scenario='RRP'):
        """
        Initialises a Gen object.

        Args:
            path (str): Directory you want to save content.

            t0 (datetime): Starting datetime.

            t1 (datetime): Ending datetime.

            region (str): Name of the pricing region of the network you want to model.

            Di (float or int): The dispatch interval. If < 1, this is interpreted as a fraction of the total number of timestamps. 
                Otherwise, it is interpreted as the number of minutes in the dispatch interval.

            Fh (float or int): The forecast horizon. If < 1, this is interpreted as a fraction of the total number of timestamps. 
                Otherwise, it is interpreted as the number of minutes in the dispatch interval.

            Fr (float): The regulation dispatch fraction. Must be < 1. 

            scenario (str): Choose a scenario from the list of scenarios embedded in this function. Must match one of the keys which 
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
        self.t0 = t0
        self.t1 = t1
        self.region = region
        self.Di = Di
        self.Fh = Fh
        self.Fr = Fr
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

        # Construct name based on dates
        name_dates = f"{dateStr}_{self.t0.strftime(selfStr)}_{self.t1.strftime(selfStr)}"

        # Construct name based on other settings
        name_settings = '_'.join(list(map(str,self.settings.values())))

        # Combine
        name = f"{name_dates}_{name_settings}"

        # Add comment
        if comment:
            name += f"_{comment}"

        return name
        

     

class BESS(Gen):
    def __init__(self,path,t0,t1,region,Di=5,Fh=0.25,Fr=0.2,scenario='RRP',Smax=2):
        """

        """
        super(BESS,self).__init__(path,t0,t1,region,Di=5,Fh=0.25,Fr=0.2,scenario='RRP')
        self.Smax = Smax
        self.settings.update(
            {
                'Smax':Smax
            }
        )

