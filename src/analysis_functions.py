# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:38:41 2020

@author: bennett.schneider
"""

import pandas as pd
import numpy as np
from error_handling import errFunc
from scipy import stats,optimize
import datetime as dt
from plotting_functions import *
import os
import time
from mip import *


def exportTimestamp(timeStr="%Y%m%d"):
    """
    Gets the current Datetime as a Pandas Datetime object and returns a formatted
    string in any desired format.

    Args:
        timeStr (str. Default=%Y%m%d): Defines the desired form of the time-stamp string in standard 
        strftime format:
            %Y -> year
            %m -> month
            %d -> day
            %H -> hour
            %M -> minute
        Default returns current time as yyyymmdd. 

    Returns:
        dateStr (str): A string giving the current datetime in the form of timeStr.
    
    Recommended use of this function is to timestamp files exported using the toolbox.

    Created on 27/02/2019 by Bennett Schneider
    """

    time = pd.Timestamp.now() # get the current timestamp
    dateStr = time.strftime(timeStr) # conver to desired format 

    return dateStr

def timeAvgEnd(data,in_freq,out_freq):
    """
    Averages a dataframe based on a period-ending convention.
    Args:
        data (pandas DataFrame): Any pandas dataframe with a datetimeindex as the index.

        freq_in (int): Frequency of the input dataframe in minutes.

        freq_out (int): Frequency of the output dataframe in minutes.

    Returns:
        data (pandas DataFrame): As input, but with all columns averaged from freq_in to freq_out.

    Created on 26/01/2021 by Bennett Schneider

    """

    data = data.copy()
    data.index = data.index - pd.Timedelta(in_freq,unit='m') # shift back by one timestep
    data = data.resample(f"{out_freq}T").mean() # average to the desired freq
    data.index = data.index + pd.Timedelta(out_freq,unit='m') # shift forward by one timestep

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

# def BESS_COINOR(rrp,freq=30,sMax=4,st0=2,eta=0.8,rMax=1,write=False,debug=True):
#     """
#     Uses the mip COIN-OR linear solver to perform a linear optimisation of a battery energy storage system (BESS)
#     dispatch strategy, based on a price curve and BESS energy storage capacity and efficiency.
    
#     Args:
#         rrp (pandas Series): Index should be a DatetimeIndex (although it doesn't HAVE to be). Values must be market prices in $/MWh.
        
#         freq (int. Default=30): Frequency of your rrp time-series in minutes.
        
#         sMax (float. Default=4): Maximum useable storage capacity of your BESS in hours.
        
#         st0 (float. Default=2): Starting capacity of your BESS in hours.
        
#         eta (float. Default=0.8): Round-trip efficiency of your BESS.
        
#         rMax (float. Default=1): Maximum rate of discharge or charge (MW). Default value of 1 essentially yields results in /MW terms.
#             This value merely scales up the results and is convenient for unit purposes. It does not affect the optimisation.
        
#         write (bool or str. Default=False): If False, does nothing. Otherwise, should be a string with a file path ending in lp.
#             This will let the function write the results of the optimisation to that location in lp format.
            
#         debug (bool. Default=True): If True, prints out messages that are useful for debugging. 
    
#     Returns:
#         results (pandas DataFrame): Index matches that of rrp. Columns are:
#           - dt_MW -> Discharge rate in MW. Charging presented as a negative value.
#           - st_MWh -> State of charge in MWh
#           - Profit_$ -> Merchant profit earned in each timestamp
#           - RRP_$/MWh -> Regional Reference Price of the given region.
    
#     Use this function to determine the optimal bidding strategy for a price-taking BESS over a given forecasted time-horizon,
#     as denoted by rrp. i.e. to calculate the optimal arbitrage revenue.
    
#     Created on 14/04/2020 by Bennett Schneider
        
#     """
    
#     rrpList = list(rrp) # make rrp a list so can use in optimiser
    
#     hr_frac = freq/60 # convert freq to fraction of hours
    
#     n = len(rrp) # Get the length of the optimisation
    
#     # Initialise the maximisation model
#     m = Model(sense='MAX')
    
#     #########################################
#     ############# Add variables #############
#     #########################################
#     # ct = [m.add_var(name=f'ct_{i}',lb=-1,ub=0) for i in range(n)] # charging rate at time, t. Ratio of max charge rate (MW)
#     dt = [m.add_var(name=f'dt_{i}',lb=-1,ub=1) for i in range(n)] # discharging rate at time, t. Ratio of max charge rate (MW)
#     # bin = [m.add_var(name=f'bin_{i}',var_type=BINARY) for i in range(n)] # Charge/discharge binary
#     st = [m.add_var(name=f'st_{i}',lb=0,ub=sMax) for i in range(n)] # storage level at time, t. Hours

#     #########################################
#     ############ Add constraints ############
#     #########################################
#     for i in range(1,n):
#         # If discharging, we lose extra capacity due to inefficiency, compared to what is actually output, but the solution becomes non-linear if we add
#         # such a constraint.
#         # Instead, embed the round trip efficiency into the storage cap. Essentially storage level represents the 'available' energy to be exported
#         m += st[i] - st[i-1] + hr_frac*eta*dt[i] == 0, f'storage_level_{i}' 

#         # Other things I tried but did not work!
#         # When bin[i] = 0, we are charging. When bin[i] = 1, we are discharging!
#         # m += st[i] - st[i-1] + hr_frac*(eta_oneWay*ct[i] + (1/eta_oneWay)*dt[i]) == 0, f'storage_level_{i}'
#         # m += st[i] - st[i-1] + eta*hr_frac*(1*ct[i] + 1*dt[i]) == 0, f'storage_level_{i}'
#         # m += st[i] - st[i-1] + hr_frac*(eta_oneWay*dt[i]*(bin[i] - 1) + (1/eta_oneWay)*dt[i]*bin[i]) == 0, f'storage_level_{i}'
#         # m += st[i] - st[i-1] + hr_frac*(eta_oneWay*dt[i] + (1/eta_oneWay)*dt[i]) == 0, f'storage_level_{i}'

#     m += st[0] == st0, 'initial_condition'
     
#     #########################################
#     ############## Objective ################
#     #########################################
#     m.objective = xsum(rrpList[i]*dt[i] for i in range(n))
    
#     #########################################
#     ############## Optimise #################
#     #########################################
#     m.optimize()
#     if debug:
#         print(f'status: {m.status} \nprofit {m.objective_value}')
    
#     if write:
#         m.write(write)

#     #########################################
#     ############ Gather Results #############
#     #########################################
#     results = pd.DataFrame(index=rrp.index)
#     res_dict = {r'dt_MW':dt,r'st_MWh':st}
#     res_dataDict = {}
    
#     for key,var in res_dict.items():
#         myVar = [v.x for v in var]
#         res_dataDict[key] = myVar
        
#     results = pd.DataFrame(res_dataDict,index=rrp.index)*rMax # scale by rMax

#     results = pd.concat([results,rrp.to_frame(r'RRP_$/MWh')],axis=1)
#     results[r'Profit_$'] = results[r'RRP_$/MWh']*results[r'dt_MW']*hr_frac
    
#     return results

# def BESS_COINOR2(rrp,m,freq=30,sMax=4,st0=2,eta=0.8,rMax=1,regDisFrac=0.2,write=False,debug=True):
#     """
#     Uses the mip COIN-OR linear solver to perform a linear optimisation of a battery energy storage system (BESS)
#     dispatch strategy, based on the energy and FCAS price curves, BESS energy storage capacity, and efficiency. 
    
#     Args:
#         rrp (pandas DataFrame): Index should be a DatetimeIndex (although it doesn't HAVE to be). Columns are the names of each of the energy and FCAS
#             markets, as shown in the AEMO tables. Values for RRP must be market prices in $/MWh. Values for FCAS must be in $/MW/hr.

#         m (MIP model): An empty model object.
        
#         freq (int. Default=30): Frequency of your rrp time-series in minutes.
        
#         sMax (float. Default=4): Maximum useable storage capacity of your BESS in hours.
        
#         st0 (float. Default=2): Starting capacity of your BESS in hours.
        
#         eta (float. Default=0.8): Round-trip efficiency of your BESS.
        
#         rMax (float. Default=1): Maximum rate of discharge or charge (MW). Default value of 1 essentially yields results in /MW terms.
#             This value merely scales up the results and is convenient for unit purposes. It does not affect the optimisation.

#         regDisFrac (float. Default=0.2): Fraction of enabled FCAS reg that is assumed to be dispatched. More research needed on this value.
        
#         write (bool or str. Default=False): If False, does nothing. Otherwise, should be a string with a file path ending in lp.
#             This will let the function write the results of the optimisation to that location in lp format.
            
#         debug (bool. Default=True): If True, prints out messages that are useful for debugging. 
    
#     Returns:
#         results (pandas DataFrame): Index matches that of rrp. Columns are:
#           - dt_MW -> Discharge rate in MW. Charging presented as a negative value.
#           - st_MWh -> State of charge in MWh
#           - Profit_$ -> Merchant profit earned in each timestamp
#           - RRP_$/MWh -> Regional Reference Price of the given region.
    
#     Use this function to determine the optimal bidding strategy for a price-taking BESS over a given forecasted time-horizon,
#     as denoted by rrp.
    
#     Created on 14/04/2020 by Bennett Schneider
        
#     """

#     # Define the key constraining constants based on rmax
#     maxAvail = 2                                      # Effective RReg FCAS MaxAvail
#     enablementMax = 1                                   # Enablement Max. Assume same for all FCAS
#     enablementMin = -1                                  # Enablement Min. Assume same for all FCAS
#     lowBreakpoint = 1                                   # Low breakpoint
#     highBreakpoint = -1                                 # High breakpoint
#     lowerSlope = (lowBreakpoint - enablementMin)/maxAvail  # Upper/Lower Slope Coeff
#     upperSlope = (enablementMax - highBreakpoint)/maxAvail # Upper/Lower Slope Coeff

#     # make rrp a list so can use in optimiser
#     enRRP = list(rrp['RRP']) # Energy rrp
#     regRaiseRRP = list(rrp['RAISEREGRRP'])
#     regLowerRRP = list(rrp['LOWERREGRRP'])
#     contRaiseRRP = list(rrp[['RAISE6SECRRP','RAISE60SECRRP','RAISE5MINRRP']].sum(axis=1)) # Sum of the contingency raise market
#     contLowerRRP = list(rrp[['LOWER6SECRRP','LOWER60SECRRP','LOWER5MINRRP']].sum(axis=1)) # Sum of the contingency lower market
    
#     hr_frac = freq/60 # convert freq to fraction of hours
    
#     n = len(rrp) # Get the length of the optimisation
    
#     #########################################
#     ############# Add variables #############
#     #########################################

#     dt = [m.add_var(name=f'dt_{i}',lb=-1,ub=1) for i in range(n)] # discharging rate at time, t. Ratio of max charge rate (MW)

#     regDt = [m.add_var(name=f'regDt_{i}',lb=-1,ub=1) for i in range(n)] # Dispatched FCAS reg discharge rate, t. Ratio of max charge rate (MW)

#     st = [m.add_var(name=f'st_{i}',lb=0,ub=sMax) for i in range(n)] # storage level at time, t. Hours
    
#     Reg_Rt = [m.add_var(name=f'Reg_Rt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Regulation FCAS raise at time, t.
#     Reg_Lt = [m.add_var(name=f'Reg_Lt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Regulation FCAS lower at time, t.

#     Cont_Rt = [m.add_var(name=f'Cont_Rt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Contingency FCAS raise at time, t.
#     Cont_Lt = [m.add_var(name=f'Cont_Lt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Contingency FCAS lower at time, t.


    
#     #########################################
#     ############ Add constraints ############
#     #########################################
#     for i in range(1,n):
#         # Force dispatch commands to 0 if their market is not represented to avoid perverse storage behaviour
#         if sum(enRRP) == 0:
#             m += dt[i] == 0, f'energy_market_{i}'

#         if sum(regRaiseRRP) == 0:
#             m += Reg_Rt[i] == 0, f'RegRaise_market_{i}'

#         if sum(regLowerRRP) == 0:
#             m += Reg_Lt[i] == 0, f'RegLower_market_{i}'

#         if sum(contRaiseRRP) == 0:
#             m += Cont_Rt[i] == 0, f'ContRaise_market_{i}'

#         if sum(contLowerRRP) == 0:
#             m += Cont_Lt[i] == 0, f'ContLower_market_{i}'

#         # Fraction of FCAS reg raise/lower that is dispatched. Calculate as a ratio between what is charged and discharged such that regDisFrac*Raise is dispatched if all raise,
#         # regDisFrac*Lower if all lower, and 0 if Raise = Lower. Linear interp in between
#         m += regDt[i] == regDisFrac*(Reg_Rt[i] - Reg_Lt[i]), f'reg_dispatch_{i}'

#         # If discharging, we lose extra capacity due to inefficiency, compared to what is actually output, but the solution becomes non-linear if we add
#         # such a constraint.
#         # Instead, embed the round trip efficiency into the storage cap. Essentially storage level represents the 'available' energy to be exported

#         # m += st[i] - st[i-1] + eta*(hr_frac*(dt[i] + regDt[i])) == 0, f'storage_level_{i}' # Assume enabled FCAS contingency never gets dispatched, but that reg does, effectively regDisFrac% of the time
#         m += st[i] - st[i-1] + eta*hr_frac*(dt[i] + regDt[i]) == 0, f'storage_level_{i}' # The optimiser abuses the power of having a deterministic regDt. Accordingly, ensure regDisFrac is set small.

#         # Apply FCAS reg raise constraints. Assume all cont. raise are always bid the same. Neglect ramp constraints. Therefore the following apply:
#         # # m += Reg_Rt[i] <= maxAvail, f"RegRaise_cap_{i}"                                             # Effective RReg FCAS MaxAvail
#         m += Reg_Rt[i] <= (enablementMax - dt[i])/upperSlope, f"Upper_RegRaise_cap_{i}"             # (Effective RReg EnablementMax - Energy Target) / Upper Slope Coeff RReg
#         # # m += Reg_Rt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_RegRaise_cap_{i}"             # (Energy Target - Effective RReg EnablementMin) / Lower Slope Coeff RReg. (Lower slope is undefined for raise)
#         m += Reg_Rt[i] <= (enablementMax - dt[i] - upperSlope*Cont_Rt[i]), f"RegRaise_jointCap_{i}" # Offer Cont. Raise EnablementMax - Energy Target - (Upper Slope Coeff Cont. Raise x Cont. Raise Target)
#         # # m += Reg_Rt[i] >= 0, f"RegRaise_floor_{i}"                                                  # if < 0, then it's reg lower, not raise
#         # m += st[i] - ((Reg_Rt[i] + dt[i])/eta)*hr_frac >= 0,f"RegRaise_storeCap_{i}"                         # Must always have enough storage to be dispatched for the entire period

#         # Apply FCAS reg lower constraints. Assume all cont. raise are always bid the same. Neglect ramp constraints. Therefore the following apply:
#         # # m += Reg_Lt[i] <= maxAvail, f"RegLower_cap_{i}"                                              # Effective LReg FCAS MaxAvail
#         # # m += Reg_Lt[i] <= (enablementMax + dt[i])/upperSlope, f"Upper_RegLower_cap_{i}"              # (Effective LReg EnablementMax - Energy Target) / Upper Slope Coeff LReg. (Upper slope is undefined for lower)
#         m += Reg_Lt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_RegLower_cap_{i}"             # (Energy Target - Effective LReg EnablementMin) / Lower Slope Coeff LReg
#         m += Reg_Lt[i] <= (-enablementMin + dt[i] - lowerSlope*Cont_Lt[i]), f"RegLower_jointCap_{i}"  # Offer Cont. Lower EnablementMax - Energy Target - (Lower Slope Coeff Cont. Raise x Cont. Lower Target)
#         # # m += Reg_Lt[i] >= 0, f"RegLower_floor_{i}"                                                   # if < 0, then it's reg raise, not lower
#         # m += st[i] + (Reg_Lt[i] - dt[i])*hr_frac*eta <= sMax,f"RegLower_storeCap_{i}"                       # Must always have enough storage to be dispatched for the entire period

#         # Apply FCAS contingecy raise constraints. Assume all cont. raise are always bid the same. Therefore the following apply:
#         # # m += Cont_Rt[i] <= maxAvail, f"ContRaise_cap_{i}"                                   # Effective RCont FCAS MaxAvail
#         m += Cont_Rt[i] <= (enablementMax - dt[i])/upperSlope, f"Upper_ContRaise_cap_{i}"   # (Offer Rxx EnablementMax - Energy Target) / Upper Slope Coeff Rxx
#         # # m += Cont_Rt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_ContRaise_cap_{i}"   # (Energy Target - Offer Rxx EnablementMin) / Lower Slope Coeff Rxx. (Lower slope is undefined for raise)
#         # # m += Cont_Rt[i] <= (enablementMax - dt[i] - Reg_Rt[i])/upperSlope                 # (Offer Rxx EnablementMax - Energy Target - RReg Target) / Upper Slope Coeff Rxx # replicated above
#         # # m += Cont_Rt[i] >= 0, f"ContRaise_floor_{i}"                                        # if < 0, then it's cont lower, not raise
#         # m += st[i] - ((Cont_Rt[i] + dt[i])/eta)*hr_frac >= 0,f"ContRaise_storeCap_{i}"               # Must always have enough storage to be dispatched for the entire period

#         # Apply FCAS contingecy lower constraints. Assume all cont. raise are always bid the same. Therefore the following apply:
#         # # m += Cont_Lt[i] <= maxAvail, f"ContLower_cap_{i}"                                   # Effective LCont FCAS MaxAvail
#         # # m += Cont_Lt[i] <= (enablementMax + dt[i])/upperSlope, f"Upper_ContLower_cap_{i}"   # (Offer Lxx EnablementMax - Energy Target) / Upper Slope Coeff Lxx. (Upper slope is undefined for
#         #  lower)
#         m += Cont_Lt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_ContLower_cap_{i}"  # (Energy Target - Offer Lxx EnablementMin) / Lower Slope Coeff Lxx
#         # # m += Cont_Lt[i] <= (enablementMax + dt[i] - Reg_Lt[i])/lowerSlope                 # (Offer Lxx EnablementMax - Energy Target - LReg Target) / Lower Slope Coeff Lxx # replicated above
#         # # m += Cont_Lt[i] >= 0, f"ContLower_floor_{i}"                                        # if < 0, then it's cont raise, not lower
#         # m += st[i] + (Cont_Lt[i] - dt[i])*hr_frac*eta <= sMax,f"ContLower_storeCap_{i}"            # Must always have enough storage to be dispatched for the entire period
        

#         # Joint storage capacity constraint (energy, reg, and cont).
#         m += st[i] >= (hr_frac/eta)*((Reg_Rt[i] + dt[i]) + 2*(Cont_Rt[i])), f"Raise_jointStorageCap_{i}" # assume that energy, reg, and contingency raise are all dispatched at one. Cont can last up to 10mins, not 5
#         m += sMax >= st[i] + eta*hr_frac*((Reg_Lt[i] - dt[i]) + 2*Cont_Lt[i]), f"Lower_jointStorageCap_{i}" # same as above, reverse signs for reg an cont, but keep same sign for dt



#     m += st[0] == st0, 'initial_condition'
    
#     #########################################
#     ############## Objective ################
#     #########################################
#     # This first, commented-out constraint includes the revenue from FCAS Reg, which are never guaranteed. So we should definitely not optimise for it (even though we do consider it in
#     # our energy balance)
#     # m.objective = xsum(rMax*(enRRP[i]*(dt[i] + regDt[i]) + Reg_Rt[i]*regRaiseRRP[i] + Reg_Lt[i]*regLowerRRP[i] + Cont_Rt[i]*contRaiseRRP[i] + Cont_Lt[i]*contLowerRRP[i]) for i in range(n))
#     m.objective = xsum(rMax*(enRRP[i]*(dt[i] + regDt[i]) + Reg_Rt[i]*regRaiseRRP[i] + Reg_Lt[i]*regLowerRRP[i] + Cont_Rt[i]*contRaiseRRP[i] + Cont_Lt[i]*contLowerRRP[i]) for i in range(n))
    
#     #########################################
#     ############## Optimise #################
#     #########################################
#     m.optimize()
#     if debug:
#         print(f'status: {m.status} \nprofit {m.objective_value}')
    
#     if write:
#         m.write(write)

#     #########################################
#     ############ Gather Results #############
#     #########################################
#     results = pd.DataFrame(index=rrp.index)
#     res_dict = {'dt_net_MW':[d + r for d,r in zip(dt,regDt)],'dt_MW':dt,'regDt_MW':regDt,'st_MWh':st,'Reg_Rt_MW':Reg_Rt,'Reg_Lt_MW':Reg_Lt,'Cont_Rt_MW':Cont_Rt,'Cont_Lt_MW':Cont_Lt}
#     res_dataDict = {}
    
#     for key,var in res_dict.items():
#         myVar = [v.x for v in var]
#         res_dataDict[key] = myVar
        
#     results = pd.DataFrame(res_dataDict,index=rrp.index) 

#     results = pd.concat([rMax*results,rrp],axis=1)
    
#     # Splits the profit out by raise/lower Reg/Cont markets
#     for col in rrp.columns:
#         if col == 'RRP':
#             varStr = 'dt_net_MW'
#         elif 'LOWER' in col:
#             if 'REG' in col:
#                 varStr = 'Reg_Lt_MW'
#             else:
#                 varStr = 'Cont_Lt_MW'
#         elif 'RAISE' in col:
#             if 'REG' in col:
#                 varStr = 'Reg_Rt_MW'
#             else:
#                 varStr = 'Cont_Rt_MW'

#         results[rf'{col}_Profit_$'] = results[col]*results[varStr]*hr_frac # half hour settlements, so multiply all profits by hr_frac
    
#     return results


def BESS_COINOR(rrp,m,freq=30,sMax=4,st0=2,eta=0.8,rMax=1,regDisFrac=0.2,regDict={1:0.2},write=False,debug=True,rrp_mod=None):
    """
    Uses the mip COIN-OR linear solver to perform a linear optimisation of a battery energy storage system (BESS)
    dispatch strategy, based on the energy and FCAS price curves, BESS energy storage capacity, and efficiency. 
    Version 3 allows a weighted average of FCAS regulation dispatch fractions to factor into the calculation.
    
    Args:
        rrp (pandas DataFrame): Index should be a DatetimeIndex (although it doesn't HAVE to be). Columns are the names of each of the energy and FCAS
            markets, as shown in the AEMO tables. Values for RRP must be market prices in $/MWh. Values for FCAS must be in $/MW/hr.

        m (MIP model): An empty model object.
        
        freq (int. Default=30): Frequency of your rrp time-series in minutes.
        
        sMax (float. Default=4): Maximum useable storage capacity of your BESS in hours.
        
        st0 (float. Default=2): Starting capacity of your BESS in hours.
        
        eta (float. Default=0.8): One-way efficiency of your BESS.
        
        rMax (float. Default=1): Maximum rate of discharge or charge (MW). Default value of 1 essentially yields results in /MW terms.
            This value merely scales up the results and is convenient for unit purposes. It does not affect the optimisation.

        regDisFrac (float. Default=0.2): Fraction of enabled FCAS reg that is assumed to be dispatched. More research needed on this value. 
        
        regDict (dict of floats. Default={1:0.2}: Probability distribution of fraction of enabled FCAS reg that is assumed to be dispatched.
            Values should be a probability such that the sum of the keys = 1. The keys should the value of regDisFrac corresponding to each probability.
        
        write (bool or str. Default=False): If False, does nothing. Otherwise, should be a string with a file path ending in lp.
            This will let the function write the results of the optimisation to that location in lp format.
            
        debug (bool. Default=True): If True, prints out messages that are useful for debugging. 

        rrp_mod (pandas DataFrame. Default=None): If a pandas dataframe is entered here, uses this for running the optimisation instead of rrp, but use rrp
            to evaluate the revenue.
    
    Returns:
        results (pandas DataFrame): Index matches that of rrp. Columns are:
          - dt_MW -> Discharge rate in MW. Charging presented as a negative value.
          - st_MWh -> State of charge in MWh
          - Profit_$ -> Merchant profit earned in each timestamp
          - RRP_$/MWh -> Regional Reference Price of the given region.
    
    Use this function to determine the optimal bidding strategy for a price-taking BESS over a given forecasted time-horizon,
    as denoted by rrp.
    
    Created on 14/04/2020 by Bennett Schneider
        
    """

    # Define the key constraining constants based on rmax
    maxAvail = 2                                      # Effective RReg FCAS MaxAvail
    enablementMax = 1                                   # Enablement Max. Assume same for all FCAS
    enablementMin = -1                                  # Enablement Min. Assume same for all FCAS
    lowBreakpoint = 1                                   # Low breakpoint
    highBreakpoint = -1                                 # High breakpoint
    lowerSlope = (lowBreakpoint - enablementMin)/maxAvail  # Upper/Lower Slope Coeff
    upperSlope = (enablementMax - highBreakpoint)/maxAvail # Upper/Lower Slope Coeff


    if (type(rrp_mod) == pd.core.frame.DataFrame) and (len(rrp_mod) > 0):
        optRRP = rrp_mod.copy()
    else:
        optRRP = rrp.copy()

    # make rrp a list so can use in optimiser
    enRRP = list(optRRP['Energy']) # Energy rrp
    regRaiseRRP = list(optRRP['RAISEREG'])
    regLowerRRP = list(optRRP['LOWERREG'])
    contRaiseRRP = list(optRRP[['RAISE6SEC','RAISE60SEC','RAISE5MIN']].sum(axis=1)) # Sum of the contingency raise market
    contLowerRRP = list(optRRP[['LOWER6SEC','LOWER60SEC','LOWER5MIN']].sum(axis=1)) # Sum of the contingency lower market
    
    hr_frac = freq/60 # convert freq to fraction of hours
    
    n = len(optRRP) # Get the length of the optimisation
    
    #########################################
    ############# Add variables #############
    #########################################

    dt = [m.add_var(name=f'dt_{i}',lb=-1,ub=1) for i in range(n)] # discharging rate at time, t. Ratio of max charge rate (MW)
    
    # collect a dictionary of possible reg discharge values (a distribution)
    regDtDist = {}
    for key,val in regDict.items():
        regDtDist[key] = [m.add_var(name=f'regDt_dist_{i}',lb=-1,ub=1) for i in range(n)] # Dispatched FCAS reg discharge rate, t. Ratio of max charge rate (MW)
    
    regDt = [m.add_var(name=f'regDt_{i}',lb=-1,ub=1) for i in range(n)] # Dispatched FCAS reg discharge rate, t. Ratio of max charge rate (MW)

    st = [m.add_var(name=f'st_{i}',lb=0,ub=sMax) for i in range(n)] # storage level at time, t. Hours
    
    Reg_Rt = [m.add_var(name=f'Reg_Rt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Regulation FCAS raise at time, t.
    Reg_Lt = [m.add_var(name=f'Reg_Lt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Regulation FCAS lower at time, t.

    Cont_Rt = [m.add_var(name=f'Cont_Rt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Contingency FCAS raise at time, t.
    Cont_Lt = [m.add_var(name=f'Cont_Lt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Contingency FCAS lower at time, t.


    
    #########################################
    ############ Add constraints ############
    #########################################
    for i in range(1,n):
        # Force dispatch commands to 0 if their market is not represented to avoid perverse storage behaviour
        if sum(enRRP) == 0:
            m += dt[i] == 0, f'energy_market_{i}'

        if sum(regRaiseRRP) == 0:
            m += Reg_Rt[i] == 0, f'RegRaise_market_{i}'

        if sum(regLowerRRP) == 0:
            m += Reg_Lt[i] == 0, f'RegLower_market_{i}'

        if sum(contRaiseRRP) == 0:
            m += Cont_Rt[i] == 0, f'ContRaise_market_{i}'

        if sum(contLowerRRP) == 0:
            m += Cont_Lt[i] == 0, f'ContLower_market_{i}'

        # Fraction of FCAS reg raise/lower that is dispatched. Calculate as a ratio between what is charged and discharged such that regDisFrac*Raise is dispatched if all raise,
        # regDisFrac*Lower if all lower, and 0 if Raise = Lower. Linear interp in between
        m += regDt[i] == regDisFrac*(Reg_Rt[i] - Reg_Lt[i]), f'reg_dispatch_{i}'
        
        for rdfrac,prob in regDict.items():
            # breakpoint()
            m += regDtDist[rdfrac][i] == rdfrac*(Reg_Rt[i] - Reg_Lt[i]), f'reg_dispatch_{rdfrac}_{i}' # regulation dispatch under different regulation dispatch fraction scenarios

        # If discharging, we lose extra capacity due to inefficiency, compared to what is actually output, but the solution becomes non-linear if we add
        # such a constraint.
        # Instead, embed the round trip efficiency into the storage cap. Essentially storage level represents the 'available' energy to be exported

        m += st[i] - st[i-1] + eta*hr_frac*(dt[i] + regDt[i]) == 0, f'storage_level_{i}' # The optimiser abuses the power of having a deterministic regDt. Accordingly, ensure regDisFrac is set small.
        # Dt = (abs(dt[i] + regDt[i]) + (dt[i] + regDt[i]))/2
        # Ct = (-abs(dt[i] + regDt[i]) + (dt[i] + regDt[i]))/2
        # m += st[i] - st[i-1] - hr_frac*(eta*Ct + (1/eta)*Dt) == 0, f'storage_level_{i}' # The optimiser abuses the power of having a deterministic regDt. Accordingly, ensure regDisFrac is set small.

        # Apply FCAS reg raise constraints. Assume all cont. raise are always bid the same. Neglect ramp constraints. Therefore the following apply:
        # # m += Reg_Rt[i] <= maxAvail, f"RegRaise_cap_{i}"                                             # Effective RReg FCAS MaxAvail
        m += Reg_Rt[i] <= (enablementMax - dt[i])/upperSlope, f"Upper_RegRaise_cap_{i}"             # (Effective RReg EnablementMax - Energy Target) / Upper Slope Coeff RReg
        # # m += Reg_Rt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_RegRaise_cap_{i}"             # (Energy Target - Effective RReg EnablementMin) / Lower Slope Coeff RReg. (Lower slope is undefined for raise)
        m += Reg_Rt[i] <= (enablementMax - dt[i] - upperSlope*Cont_Rt[i]), f"RegRaise_jointCap_{i}" # Offer Cont. Raise EnablementMax - Energy Target - (Upper Slope Coeff Cont. Raise x Cont. Raise Target)
        # # m += Reg_Rt[i] >= 0, f"RegRaise_floor_{i}"                                                  # if < 0, then it's reg lower, not raise
        # m += st[i] - ((Reg_Rt[i] + dt[i])/eta)*hr_frac >= 0,f"RegRaise_storeCap_{i}"                         # Must always have enough storage to be dispatched for the entire period

        # Apply FCAS reg lower constraints. Assume all cont. raise are always bid the same. Neglect ramp constraints. Therefore the following apply:
        # # m += Reg_Lt[i] <= maxAvail, f"RegLower_cap_{i}"                                              # Effective LReg FCAS MaxAvail
        # # m += Reg_Lt[i] <= (enablementMax + dt[i])/upperSlope, f"Upper_RegLower_cap_{i}"              # (Effective LReg EnablementMax - Energy Target) / Upper Slope Coeff LReg. (Upper slope is undefined for lower)
        m += Reg_Lt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_RegLower_cap_{i}"             # (Energy Target - Effective LReg EnablementMin) / Lower Slope Coeff LReg
        m += Reg_Lt[i] <= (-enablementMin + dt[i] - lowerSlope*Cont_Lt[i]), f"RegLower_jointCap_{i}"  # Offer Cont. Lower EnablementMax - Energy Target - (Lower Slope Coeff Cont. Raise x Cont. Lower Target)
        # # m += Reg_Lt[i] >= 0, f"RegLower_floor_{i}"                                                   # if < 0, then it's reg raise, not lower
        # m += st[i] + (Reg_Lt[i] - dt[i])*hr_frac*eta <= sMax,f"RegLower_storeCap_{i}"                       # Must always have enough storage to be dispatched for the entire period

        # Apply FCAS contingecy raise constraints. Assume all cont. raise are always bid the same. Therefore the following apply:
        # # m += Cont_Rt[i] <= maxAvail, f"ContRaise_cap_{i}"                                   # Effective RCont FCAS MaxAvail
        m += Cont_Rt[i] <= (enablementMax - dt[i])/upperSlope, f"Upper_ContRaise_cap_{i}"   # (Offer Rxx EnablementMax - Energy Target) / Upper Slope Coeff Rxx
        # # m += Cont_Rt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_ContRaise_cap_{i}"   # (Energy Target - Offer Rxx EnablementMin) / Lower Slope Coeff Rxx. (Lower slope is undefined for raise)
        # # m += Cont_Rt[i] <= (enablementMax - dt[i] - Reg_Rt[i])/upperSlope                 # (Offer Rxx EnablementMax - Energy Target - RReg Target) / Upper Slope Coeff Rxx # replicated above
        # # m += Cont_Rt[i] >= 0, f"ContRaise_floor_{i}"                                        # if < 0, then it's cont lower, not raise
        # m += st[i] - ((Cont_Rt[i] + dt[i])/eta)*hr_frac >= 0,f"ContRaise_storeCap_{i}"               # Must always have enough storage to be dispatched for the entire period

        # Apply FCAS contingecy lower constraints. Assume all cont. raise are always bid the same. Therefore the following apply:
        # # m += Cont_Lt[i] <= maxAvail, f"ContLower_cap_{i}"                                   # Effective LCont FCAS MaxAvail
        # # m += Cont_Lt[i] <= (enablementMax + dt[i])/upperSlope, f"Upper_ContLower_cap_{i}"   # (Offer Lxx EnablementMax - Energy Target) / Upper Slope Coeff Lxx. (Upper slope is undefined for
        #  lower)
        m += Cont_Lt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_ContLower_cap_{i}"  # (Energy Target - Offer Lxx EnablementMin) / Lower Slope Coeff Lxx
        # # m += Cont_Lt[i] <= (enablementMax + dt[i] - Reg_Lt[i])/lowerSlope                 # (Offer Lxx EnablementMax - Energy Target - LReg Target) / Lower Slope Coeff Lxx # replicated above
        # # m += Cont_Lt[i] >= 0, f"ContLower_floor_{i}"                                        # if < 0, then it's cont raise, not lower
        # m += st[i] + (Cont_Lt[i] - dt[i])*hr_frac*eta <= sMax,f"ContLower_storeCap_{i}"            # Must always have enough storage to be dispatched for the entire period
        

        # Joint storage capacity constraint (energy, reg, and cont).
        m += st[i] >= (hr_frac/eta)*((Reg_Rt[i] + dt[i]) + 2*(Cont_Rt[i])), f"Raise_jointStorageCap_{i}" # assume that energy, reg, and contingency raise are all dispatched at one. Cont can last up to 10mins, not 5
        m += sMax >= st[i] + eta*hr_frac*((Reg_Lt[i] - dt[i]) + 2*Cont_Lt[i]), f"Lower_jointStorageCap_{i}" # same as above, reverse signs for reg an cont, but keep same sign for dt



    m += st[0] == st0, 'initial_condition'
    
    #########################################
    ############## Objective ################
    #########################################
    # This first, commented-out constraint includes the revenue from FCAS Reg, which are never guaranteed. So we should definitely not optimise for it (even though we do consider it in
    # our energy balance)
    # m.objective = xsum(rMax*(enRRP[i]*(dt[i] + regDt[i]) + Reg_Rt[i]*regRaiseRRP[i] + Reg_Lt[i]*regLowerRRP[i] + Cont_Rt[i]*contRaiseRRP[i] + Cont_Lt[i]*contLowerRRP[i]) for i in range(n))

    # breakpoint()

    m.objective = xsum(
            (
            sum(
                [regDict[regDis]*(enRRP[i]*(dt[i] + regDis_i[i]) + Reg_Rt[i]*regRaiseRRP[i] + Reg_Lt[i]*regLowerRRP[i] + Cont_Rt[i]*contRaiseRRP[i] + Cont_Lt[i]*contLowerRRP[i]) for regDis,regDis_i in regDtDist.items()]
                )
            ) for i in range(n)
        )
    
    #########################################
    ############## Optimise #################
    #########################################
    m.optimize()
    if debug:
        print(f'status: {m.status} \nprofit {m.objective_value}')
    
    if write:
        m.write(write)

    #########################################
    ############ Gather Results #############
    #########################################

    # if type(rrp_mod) == pd.core.frame.DataFrame:
    #     rrp = rrp_mod.copy()
    # else:
    #     rrp = rrp.copy()

    results = pd.DataFrame(index=rrp.index)
    # res_dict = {'dt_net_MW':[d + r for d,r in zip(dt,regDt)],'dt_MW':dt,'regDt_MW':regDt,'st_MWh':st,'Reg_Rt_MW':Reg_Rt,'Reg_Lt_MW':Reg_Lt,'Cont_Rt_MW':Cont_Rt,'Cont_Lt_MW':Cont_Lt}
    res_dict = {'Energy_MW':[d + r for d,r in zip(dt,regDt)],'dt_MW':dt,'regDt_MW':regDt,'st_MWh':st,'REGRAISE_MW':Reg_Rt,'REGLOWER_MW':Reg_Lt,'CONTRAISE_MW':Cont_Rt,'CONTLOWER_MW':Cont_Lt}
    res_dataDict = {}
    for key,var in res_dict.items():
        myVar = [v.x for v in var]
        res_dataDict[key] = myVar
    
    results = pd.DataFrame(res_dataDict,index=rrp.index)*rMax
    # results = pd.concat([rMax*results,rrp],axis=1)
    
    # Splits the profit out by raise/lower Reg/Cont markets
    for col in rrp.columns:
        
        if 'LOWER' in col:
            if 'REG' in col:
                varStr = 'REGLOWER_MW'
            else:
                varStr = 'CONTLOWER_MW'
        elif 'RAISE' in col:
            if 'REG' in col:
                varStr = 'REGRAISE_MW'
            else:
                varStr = 'CONTRAISE_MW'

        else:
            varStr = col + '_MW'

        try:
            results[f"{col}_$"] = rrp[col]*results[varStr]*hr_frac # half hour settlements, so multiply all profits by hr_frac
        except TypeError:
            pass

    return results

def BESS_COINOR_hurdle(
    rrp,
    m,
    RFr=0.2,
    LFr=0.2,
    Fr_dist=None,
    hurdle=0,
    freq=30,
    sMax=4,
    st0=2,
    eta=0.8,
    rMax=1,
    write=False,
    rrp_mod=None,
    maxAvail= 2,                  # Effective RReg FCAS MaxAvail 
    enablementMax = 1,             # Enablement Max. Assume same for all FCAS
    enablementMin = -1,            # Enablement Min. Assume same for all FCAS
    lowBreakpoint = 1,             # Low breakpoint
    highBreakpoint = -1            # High breakpoint
    ):
    """
    Uses the mip COIN-OR linear solver to perform a linear optimisation of a battery energy storage system (BESS)
    dispatch strategy, based on the energy and FCAS price curves, BESS energy storage capacity, and efficiency. 
    Version 3 allows a weighted average of FCAS regulation dispatch fractions to factor into the calculation.
    
    Args:
        rrp (pandas DataFrame): Index should be a DatetimeIndex (although it doesn't HAVE to be). Columns are the names of each of the energy and FCAS
            markets, as shown in the AEMO tables. Values for RRP must be market prices in $/MWh. Values for FCAS must be in $/MW/hr.

        m (MIP model): An empty model object.

        RFr (float. Default=0.2): Deterministic constant based on the revenue-weighted average regulation raise dispatch fraction.

        LFr (float. Default=0.2): ADeterministic constant based on the revenue-weighted average regulation lower dispatch fraction.

        Fr_dist (pandas DataFrame): Unweighted distribution of regulation raise dispatch fraction. Index is dispatch fraction, columns must be RAISE and LOWER and values are the
            respective probabilities of those fractions which will be passed to the optimiser. Probabilties expressed between 0 and 1 and each column must add to 1. Recommended indices
            are 0, 0.2, 0.4, 0.6, 0.8, 1. 

        hurdle (float. Default=0): Hurdle price given to the optimiser in $/MWh.
        
        freq (int. Default=30): Frequency of your rrp time-series in minutes.
        
        sMax (float. Default=4): Maximum useable storage capacity of your BESS in hours.
        
        st0 (float. Default=2): Starting capacity of your BESS in hours.
        
        eta (float. Default=0.8): Round trip efficiency of your BESS.
        
        rMax (float. Default=1): Maximum rate of discharge or charge (MW). Default value of 1 essentially yields results in /MW terms.
            This value merely scales up the results and is convenient for unit purposes. It does not affect the optimisation.
        
        write (bool or str. Default=False): If False, does nothing. Otherwise, should be a string with a file path ending in lp.
            This will let the function write the results of the optimisation to that location in lp format.

        rrp_mod (pandas DataFrame. Default=None): If a pandas dataframe is entered here, uses this for running the optimisation instead of rrp, but use rrp
            to evaluate the revenue.
    
    Returns:
        results (pandas DataFrame): Index matches that of rrp. Columns are:
          - dt_MW -> Discharge rate in MW. Charging presented as a negative value.
          - st_MWh -> State of charge in MWh
          - Profit_$ -> Merchant profit earned in each timestamp
          - RRP_$/MWh -> Regional Reference Price of the given region.
    
    Use this function to determine the optimal bidding strategy for a price-taking BESS over a given forecasted time-horizon,
    as denoted by rrp.
    
    Created on 14/04/2020 by Bennett Schneider
        
    """ 
    eta1 = np.sqrt(eta) # round-trip to one-way

    # Define the key constraining constants based on rmax
    lowerSlope = (lowBreakpoint - enablementMin)/maxAvail  # Upper/Lower Slope Coeff
    upperSlope = (enablementMax - highBreakpoint)/maxAvail # Upper/Lower Slope Coeff


    if (type(rrp_mod) == pd.core.frame.DataFrame) and (len(rrp_mod) > 0):
        optRRP = rrp_mod.copy()
    else:
        optRRP = rrp.copy()

    # make rrp a list so can use in optimiser
    enRRP = list(optRRP['Energy']) # Energy rrp
    regRaiseRRP = list(optRRP['RAISEREG'])
    regLowerRRP = list(optRRP['LOWERREG'])
    contRaiseRRP = list(optRRP[['RAISE6SEC','RAISE60SEC','RAISE5MIN']].sum(axis=1)) # Sum of the contingency raise market
    contLowerRRP = list(optRRP[['LOWER6SEC','LOWER60SEC','LOWER5MIN']].sum(axis=1)) # Sum of the contingency lower market
    
    hr_frac = freq/60 # convert freq to fraction of hours
    
    n = len(optRRP) # Get the length of the optimisation

    if type(Fr_dist) != pd.core.frame.DataFrame:
        if Fr_dist == None:
            Fr_dist = pd.DataFrame({'RAISE':[RFr],'LOWER':[LFr]},index=[1]) # if None, allow the objective function complete deterministic knowledge of Fr
        else:
            sys.exit("Please enter Fr_dist as either None or a pandas DataFrame, as per the documentation for analysis_functions.BESS_COINOR_hurdle()")

    #########################################
    ############# Add variables #############
    #########################################

    # dt = [m.add_var(name=f'dt_{i}',lb=-1,ub=1) for i in range(n)] # discharging rate at time, t. Ratio of max charge rate (MW)
    dt = [m.add_var(name=f'dt_{i}',lb=0,ub=1) for i in range(n)] # discharging rate at time, t. Ratio of max charge rate (MW)
    ct = [m.add_var(name=f'ct_{i}',lb=0,ub=1) for i in range(n)] # charging rate at time, t. Ratio of max charge rate (MW)

    # Binary variables that control whether the battery is charging or discharging
    bdt = [m.add_var(name=f'bdt_{i}',var_type='B') for i in range(n)] # 1 during discharge
    bct = [m.add_var(name=f'bct_{i}',var_type='B') for i in range(n)] # 1 during charge

    regDt = [m.add_var(name=f'regDt_{i}',lb=0,ub=1) for i in range(n)] # Actual dispatched FCAS reg discharge rate, t. Ratio of max charge rate (MW)
    regCt = [m.add_var(name=f'regCt_{i}',lb=0,ub=1) for i in range(n)] # Actual dispatched FCAS reg charge rate, t. Ratio of max charge rate (MW)

    st = [m.add_var(name=f'st_{i}',lb=0,ub=sMax) for i in range(n)] # storage level at time, t. Hours
    
    Reg_Rt = [m.add_var(name=f'Reg_Rt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Regulation FCAS raise at time, t.
    Reg_Lt = [m.add_var(name=f'Reg_Lt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Regulation FCAS lower at time, t.

    Cont_Rt = [m.add_var(name=f'Cont_Rt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Contingency FCAS raise at time, t.
    Cont_Lt = [m.add_var(name=f'Cont_Lt_{i}',lb=0,ub=maxAvail) for i in range(n)] # this is the MW that are available for Contingency FCAS lower at time, t.


    
    #########################################
    ############ Add constraints ############
    #########################################
    phi_t = []
    m += st[0] == st0, 'initial_condition'
    for i in range(1,n):
        # Force dispatch commands to 0 if their market is not represented to avoid perverse storage behaviour
        if sum(enRRP) == 0:
            m += dt[i] == 0, f'energy_market_{i}'

        if sum(regRaiseRRP) == 0:
            m += Reg_Rt[i] == 0, f'RegRaise_market_{i}'

        if sum(regLowerRRP) == 0:
            m += Reg_Lt[i] == 0, f'RegLower_market_{i}'

        if sum(contRaiseRRP) == 0:
            m += Cont_Rt[i] == 0, f'ContRaise_market_{i}'

        if sum(contLowerRRP) == 0:
            m += Cont_Lt[i] == 0, f'ContLower_market_{i}'

        # Apply the binary variables to constrain charge/discharge behaviour
        m += bdt[i] + bct[i] == 1
        m += dt[i] <= rMax*bdt[i] # constrain discharge
        m += ct[i] <= rMax*bct[i] # constrain charge

        # Fraction of FCAS reg raise/lower that is dispatched. Can be dispatched in both at the same time in principle
        m += regDt[i] == RFr*Reg_Rt[i], f'regraise_dispatch_{i}' # deterministic constant based on the revenue-weighted average
        m += regCt[i] == LFr*Reg_Lt[i],f'reglower_dispatch_{i}' # deterministic constant based on the revenue-weighted average

        # Net dispatched discharge and charge volumes 
        dt_net = dt[i] + regDt[i] 
        ct_net = ct[i] + regCt[i]

        # Apply the most accurate bess storage model
        m += st[i] - st[i-1] + hr_frac*(dt_net/eta1 - eta1*ct_net) == 0, f'storage_level_{i}' 

        # Apply FCAS reg raise constraints. Assume all cont. raise are always bid the same. Neglect ramp constraints. Therefore the following apply:
        m += Reg_Rt[i] <= (enablementMax - dt[i])/upperSlope, f"Upper_RegRaise_cap_{i}"             # (Effective RReg EnablementMax - Energy Target) / Upper Slope Coeff RReg
        m += Reg_Rt[i] <= (enablementMax - dt[i] - upperSlope*Cont_Rt[i]), f"RegRaise_jointCap_{i}" # Offer Cont. Raise EnablementMax - Energy Target - (Upper Slope Coeff Cont. Raise x Cont. Raise Target)

        # Apply FCAS reg lower constraints. Assume all cont. raise are always bid the same. Neglect ramp constraints. Therefore the following apply:
        m += Reg_Lt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_RegLower_cap_{i}"             # (Energy Target - Effective LReg EnablementMin) / Lower Slope Coeff LReg
        m += Reg_Lt[i] <= (-enablementMin + dt[i] - lowerSlope*Cont_Lt[i]), f"RegLower_jointCap_{i}"  # Offer Cont. Lower EnablementMax - Energy Target - (Lower Slope Coeff Cont. Raise x Cont. Lower Target)

        # Apply FCAS contingecy raise constraints. Assume all cont. raise are always bid the same. Therefore the following apply:
        m += Cont_Rt[i] <= (enablementMax - dt[i])/upperSlope, f"Upper_ContRaise_cap_{i}"   # (Offer Rxx EnablementMax - Energy Target) / Upper Slope Coeff Rxx

        # Apply FCAS contingecy lower constraints. Assume all cont. raise are always bid the same. Therefore the following apply:
        m += Cont_Lt[i] <= (dt[i] - enablementMin)/lowerSlope, f"Lower_ContLower_cap_{i}"  # (Energy Target - Offer Lxx EnablementMin) / Lower Slope Coeff Lxx
        
        # Joint storage capacity constraint (energy, reg, and cont).
        m += st[i] >= (hr_frac/eta1)*((Reg_Rt[i] + dt[i]) + 2*(Cont_Rt[i])), f"Raise_jointStorageCap_{i}" # assume that energy, reg, and contingency raise are all dispatched at one. Cont can last up to 10mins, not 5
        m += sMax >= st[i] + eta1*hr_frac*((Reg_Lt[i] - dt[i]) + 2*Cont_Lt[i]), f"Lower_jointStorageCap_{i}" # same as above, reverse signs for reg an cont, but keep same sign for dt


        ###### Construct Objective #######

        # Weighted average of the component of revenue paid by regulation dispatch into the energy market
        phi_t_n = sum([bn*(Fr_dist.loc[bn,'RAISE']*Reg_Rt[i] - Fr_dist.loc[bn,'LOWER'])*(enRRP[i] - hurdle) for bn in Fr_dist.index])

        phi_t.append(phi_t_n + Reg_Rt[i]*regRaiseRRP[i] + Reg_Lt[i]*regLowerRRP[i] + Cont_Rt[i]*contRaiseRRP[i] + Cont_Lt[i]*contLowerRRP[i] + (dt[i] - ct[i])*enRRP[i] - dt[i]*hurdle)

    # m += st[0] == st0, 'initial_condition'
    
    #########################################
    ############## Objective ################
    #########################################

    m.objective = xsum(phi_t)
    
    #########################################
    ############## Optimise #################
    #########################################
    m.optimize()
    logging.debug(f' status: {m.status} \nprofit {m.objective_value}')
    
    if write:
        m.write(write)

    #########################################
    ############ Gather Results #############
    #########################################

    # if type(rrp_mod) == pd.core.frame.DataFrame:
    #     rrp = rrp_mod.copy()
    # else:
    #     rrp = rrp.copy()

    results = pd.DataFrame(index=rrp.index)
    
    # res_dict = {'dt_net_MW':[d + r for d,r in zip(dt,regDt)],'dt_MW':dt,'regDt_MW':regDt,'st_MWh':st,'Reg_Rt_MW':Reg_Rt,'Reg_Lt_MW':Reg_Lt,'Cont_Rt_MW':Cont_Rt,'Cont_Lt_MW':Cont_Lt}
    res_dict = {'dt_MW':dt,'ct_MW':ct,'regDt_MW':regDt,'regCt_MW':regCt,'st_MWh':st,'RAISEREG_MW':Reg_Rt,'LOWERREG_MW':Reg_Lt,'RAISECONT_MW':Cont_Rt,'LOWERCONT_MW':Cont_Lt}
    res_dataDict = {}
    for key,var in res_dict.items():
        myVar = [v.x for v in var]
        res_dataDict[key] = myVar
    
    results = pd.DataFrame(res_dataDict,index=rrp.index)*rMax

    # Combine the charge and discharge results for more intuitive plotting 
    results['dt_MW'] = results['dt_MW'] - results['ct_MW']
    results['regDt_MW'] = results['regDt_MW'] - results['regCt_MW']
    
    results.drop(['ct_MW','regCt_MW'],axis=1,inplace=True) # drop charge info now it's been integrated

    # Calculate the net energy dispatch
    results['Energy_MW'] = results['dt_MW'] + results['regDt_MW']

    # results = pd.concat([rMax*results,rrp],axis=1)

    # Splits the profit out by raise/lower Reg/Cont markets
    for col in rrp.columns:
        
        if 'LOWER' in col:
            if 'REG' in col:
                varStr = 'LOWERREG_MW'
            else:
                varStr = 'LOWERCONT_MW'
        elif 'RAISE' in col:
            if 'REG' in col:
                varStr = 'RAISEREG_MW'
            else:
                varStr = 'RAISECONT_MW'

        else:
            varStr = col + '_MW'

        try:
            results[f"{col}_$"] = rrp[col]*results[varStr]*hr_frac # half hour settlements, so multiply all profits by hr_frac
        except TypeError:
            pass

    return results

def horizonDispatch(RRP,m,freq,tFcst,tInt,optfunc=BESS_COINOR,st0=2,rrp_mod=None,rMax=1,**kwargs):
    """
    A wrapper around BESS_COINOR that runs a moving forecast window of width tFcst in hours at intervals
    of tInt. E.g. 2-day forecast updated and optimised on every 30min interval.
    
    Args:
        RRP (pandas DataFrame): Index must be DatetimeIndex. Should correspond to a static price forecast of the energy market in $/MWh
            and to each of the FCAS markets in $/MW/h.
        
        m (MIP model): Empty model with sense 'MAX'.
        
        freq (int): Frequency of RRP in minutes.
        
        tFcst (int): Length of the slice of the price forecast you want to pass to the optimiser in HOURS.
        
        tInt (int): Length of the interval over which you want to implement the optimised operations before seeing
            the next section of the forecast in MINUTES.
        
        optfunc (function. Default=BESS_COINOR_hurdle): Any valid bess optimiser where the first two required arguments are rrp_frame and m
            and which has an optional argments rrp_mod, st0, rrp_mod, st0, rMax, debug, and freq.

        st0 (float. Default=2): Starting capacity of your BESS in hours.

        rrp_mod (pandas DataFrame. Default=None): If a pandas dataframe is entered here, uses this for evaluating the optimisation, but uses rrp for evaluating actual revenue.

        rMax (float. Default=1): Maximum rate of discharge or charge (MW). Default value of 1 essentially yields results in /MW terms.
            This value merely scales up the results and is convenient for unit purposes. It does not affect the optimisation.
            
        **kwargs (inputs): Any optional inputs from optfunc() except rrp_mod, st0, rMax, debug, and freq
        
    Functions used:
        analysis_functions:
            - BESS_COINOR() (default)

    Returns:
        results (pandas DataFrame): Columns are as for BESS_COINOR, but with "DailyProfit_$" added as a column. Results
            are only given for the actioned dispatches according to the set interval, not for dispatches that were optimised over the 
            forecast horizon but which did not come to fruition.
    
    Use this function to experiment with different BESS sizes/specs and forecast horizons and intervals. Works best with a predefined
    forecast, rather than one that changes every interval. Future implementations could allow a dictionary of forecasts to be passed,
    and looked up for a given time-stamp.
    
    Created on 16/04/2020 by Bennett Schneider
    
    """
    RESULTS = []
    numInts = 60/freq # number of intervals in an hour
    # for i in range(0,int(days*24*60/tInt)):

    for i in range(0,int(freq*(len(RRP))/tInt)):
        # breakpoint()
        start = int(numInts*i*tInt/60) # define start of the interval
        end = int(numInts*(i*tInt/60+tFcst)) # define end of the interval
        
        if i > 0:
            start -= 1 # go back one time-step so we get an overlap of the initial conditions
        
        rrp_frame = RRP.iloc[start:end]
        
        # Note for future (as of 14/2/21):
        # Use rrp_frame.index[0] and rrp_frame.index[-1] to 
        # determine the time slice for rrp_mod, which must be compatible 
        # with a forecasted dataset
        if type(rrp_mod) == pd.core.frame.DataFrame:
            rrp_frame_mod = rrp_mod.iloc[start:end]
        else:
            rrp_frame_mod = None
        
        logging.debug(f" Running {optfunc.__name__} between {rrp_frame.index[0]} and {rrp_frame.index[-1]}")
        # something weird happening with the inputs making rrp_frame_mod and empty dataframe instead of None
        results = optfunc(rrp_frame,m,freq=freq,rrp_mod=rrp_frame_mod,st0=st0,rMax=rMax,**kwargs)
  
        m.clear()
        
        if i > 0 :
            results = results[:int(numInts*tInt/60)+1]
            results = results[1:] # remove the first element because it's a duplicate
        else:
            results = results[:int(numInts*tInt/60)] # restrict results to those of a single dispatch interval (not the forecast length)

        # print(results)
        # breakpoint()
        st0 = results['st_MWh'].iloc[-1]/rMax # the last state of charge is now the initial condition for the next interval
        
        RESULTS.append(results)
    
    # concatenate all the results together
    results = pd.concat(RESULTS)
    
    # save the index name for later (should be 'Timestamp')
    indexName = results.index.name

    # Split results into two categories
    opsCols = ['dt_MW','regDt_MW','st_MWh'] 
    revCols = [col for col in results if col not in opsCols]
    operations = results.copy()[[col for col in opsCols if col in list(results.columns)]]
    revenue = results.copy()[revCols]

    # stack the revenue table
    # split into dispatch and revenue
    dispatch = revenue[[col for col in revenue if 'MW' in col]]
    revenue = revenue[[col for col in revenue if '$' in col]]

    # remove the identifiers from the columns
    dispatch.columns = [col.split('_')[0] for col in dispatch]
    revenue.columns = [col.split('_')[0].replace('RRP','') for col in revenue]

    # modify dispatch so we can map back 1:1 to the original markets
    for rl in ['RAISE','LOWER']:
        origCol = rl + 'CONT'
        for market in ['6SEC','60SEC','5MIN']:
            col = rl + market
            dispatch[col] = dispatch[origCol] # we dispatch the same in all lower and raise markets
        dispatch.drop(origCol,axis=1,inplace=True) # drop the aggregated cont columns

    # stack the dataframes and modify the new column names
    dispatch = dispatch.stack().reset_index().rename({'level_1':'Market',0:'Dispatch_MW'},axis=1).set_index([indexName,'Market'])
    revenue = revenue.stack().reset_index().rename({'level_1':'Market',0:'Revenue_$'},axis=1).set_index([indexName,'Market'])

    # Concatenate horizontally and set Timestamp as the sole index
    revenue = pd.concat([revenue,dispatch],axis=1).reset_index().set_index(indexName)
    
    return revenue,operations


# def horizonDispatch(RRP,m,freq,tFcst,tInt,sMax=4,st0=2,eta=0.8,rMax=1,regDisFrac=0.2,regDict={1:0.2},debug=True,rrp_mod=None):
#     """
#     A wrapper around BESS_COINOR that runs a moving forecast window of width tFcst in hours at intervals
#     of tInt. E.g. 2-day forecast updated and optimised on every 30min interval.
    
#     Args:
#         RRP (pandas DataFrame): Index must be DatetimeIndex. Should correspond to a static price forecast of the energy market in $/MWh
#             and to each of the FCAS markets in $/MW/h.
        
#         m (MIP model): Empty model with sense 'MAX'.
        
#         freq (int): Frequency of RRP in minutes.
        
#         tFcst (int): Length of the slice of the price forecast you want to pass to the optimiser in HOURS.
        
#         tInt (int): Length of the interval over which you want to implement the optimised operations before seeing
#             the next section of the forecast in MINUTES.
            
#         sMax (float. Default=4): Maximum useable storage capacity of your BESS in hours.
        
#         st0 (float. Default=2): Starting capacity of your BESS in hours.
        
#         eta (float. Default=0.8): Round-trip efficiency of your BESS.
        
#         rMax (float. Default=1): Maximum rate of discharge or charge (MW). Default value of 1 essentially yields results in /MW terms.
#             This value merely scales up the results and is convenient for unit purposes. It does not affect the optimisation.

#         regDisFrac (float. Default=0.2): Fraction of enabled FCAS reg that is assumed to be dispatched. More research needed on this value.
            
#         debug (bool. Default=True): If True, prints out messages from BESS_COINOR that are useful for debugging. 

#         rrp_mod (pandas DataFrame. Default=None): If a pandas dataframe is entered here, uses this for evaluating the optimisation, but uses rrp for evaluating actual revenue.
        
#     Functions used:
#         analysis_functions:
#             - BESS_COINOR()
    
#     Returns:
#         results (pandas DataFrame): Columns are as for BESS_COINOR, but with "DailyProfit_$" added as a column. Results
#             are only given for the actioned dispatches according to the set interval, not for dispatches that were optimised over the 
#             forecast horizon but which did not come to fruition.
    
#     Use this function to experiment with different BESS sizes/specs and forecast horizons and intervals. Works best with a predefined
#     forecast, rather than one that changes every interval. Future implementations could allow a dictionary of forecasts to be passed,
#     and looked up for a given time-stamp.
    
#     Created on 16/04/2020 by Bennett Schneider
    
#     """
#     RESULTS = []
#     numInts = 60/freq # number of intervals in an hour
#     # for i in range(0,int(days*24*60/tInt)):

#     for i in range(0,int(freq*(len(RRP))/tInt)):
#         # breakpoint()
#         start = int(numInts*i*tInt/60) # define start of the interval
#         end = int(numInts*(i*tInt/60+tFcst)) # define end of the interval
        
#         if i > 0:
#             start -= 1 # go back one time-step so we get an overlap of the initial conditions
        
#         rrp_frame = RRP.iloc[start:end]
        
#         # Note for future (as of 14/2/21):
#         # Use rrp_frame.index[0] and rrp_frame.index[-1] to 
#         # determine the time slice for rrp_mod, which must be compatible 
#         # with a forecasted dataset
#         if type(rrp_mod) == pd.core.frame.DataFrame:
#             rrp_frame_mod = rrp_mod.iloc[start:end]
#         else:
#             rrp_frame_mod = None

#         if debug:
#             print(f"Running BESS COIN OR between {rrp_frame.index[0]} and {rrp_frame.index[-1]}")

#         # write = rf"C:\Users\bennett.schneider\OneDrive - Australian National University\Master of Energy Change\SCNC8021\Analysis\FCAS\debug\{rrp_series.index[0].strftime('%Y%m%d')}.lp"
#         results = BESS_COINOR(rrp_frame,m,freq=freq,sMax=sMax,st0=st0,eta=eta,rMax=rMax,regDisFrac=regDisFrac,regDict=regDict,write=False,debug=True,rrp_mod=rrp_frame_mod)
#         # print(results)
#         # breakpoint()

#         m.clear()

        
#         if i > 0 :
#             results = results[:int(numInts*tInt/60)+1]
#             results = results[1:] # remove the first element because it's a duplicate
#         else:
#             results = results[:int(numInts*tInt/60)] # restrict results to those of a single dispatch interval (not the forecast length)

#         # print(results)
#         # breakpoint()

        
#         st0 = results['st_MWh'].iloc[-1]/rMax # the last state of charge is now the initial condition for the next interval
        
#         RESULTS.append(results)
    
#     # concatenate all the results together
#     results = pd.concat(RESULTS)

#     # save the index name for later (should be 'Timestamp')
#     indexName = results.index.name

#     # Split results into two categories
#     opsCols = ['dt_MW','regDt_MW','st_MWh'] 
#     revCols = [col for col in results if col not in opsCols]
#     operations = results.copy()[opsCols]
#     revenue = results.copy()[revCols]

#     # stack the revenue table
#     # split into dispatch and revenue
#     dispatch = revenue[[col for col in revenue if 'MW' in col]]
#     revenue = revenue[[col for col in revenue if '$' in col]]

#     # remove the identifiers from the columns
#     dispatch.columns = [col.split('_')[0] for col in dispatch]
#     revenue.columns = [col.split('_')[0].replace('RRP','') for col in revenue]

#     # modify dispatch so we can map back 1:1 to the original markets
#     for rl in ['RAISE','LOWER']:
#         origCol = 'CONT' + rl
#         for market in ['6SEC','60SEC','5MIN']:
#             col = rl + market
#             dispatch[col] = dispatch[origCol] # we dispatch the same in all lower and raise markets
#         dispatch.drop(origCol,axis=1,inplace=True) # drop the aggregated cont columns

#     # stack the dataframes and modify the new column names
#     dispatch = dispatch.stack().reset_index().rename({'level_1':'Market',0:'Dispatch_MW'},axis=1).set_index([indexName,'Market'])
#     revenue = revenue.stack().reset_index().rename({'level_1':'Market',0:'Revenue_$'},axis=1).set_index([indexName,'Market'])
#     # Concatenate horizontally and set Timestamp as the sole index
#     revenue = pd.concat([revenue,dispatch],axis=1).reset_index().set_index(indexName)

#     # # add a column for daily profit
#     # td = pd.Timedelta(minutes=freq)
#     # for col in RRP.columns:
#     #     profit = results[f'{col}_Profit_$'].copy()
#     #     profit.index = profit.index - td
#     #     profit = profit.resample('d',loffset=td).sum() 
#     #     profit = profit.reindex(results.index,method='ffill')
#     #     results[f'{col}_DailyProfit_$'] = profit
    
#     return revenue,operations

def dailyPriceBands(Region,metaVal,priceBands,sMax,freq=5,mode='quarterly'):
    """
    For a given Region object, creates a nested dict of pandas dataframes containing the daily average of the results attribute, filtered by the daily price delta, as
    indicated by the list of tuples in priceBands.
    
    Args:
        Region (Region): A Region object.
          - results (pandas DataFrame)
         
         priceBands (list of tups): Each tuple pair is a range of daily price deltas. The results attribute will be sorted accoring to these ranges.
         
         sMax (int): The maximum hours of storage for which you want to generate results.

         freq (int. Default=5): Frequency of Region.results in mins.
         
         mode (str. Default='quarterly'): Controls the periodicity of the output.
             quarterly -> dpPres keys are datetimes representing the start of a given quarter
             annual -> dpPres keys are the datetime representing the start of a given financial year
    
    Adds to Region as attribute:
        dpPres (dict of dict of pandas DataFrames): Keys are datetimes, values are the tuples in priceBands, which are in turn keys for a pandas DataFrame which is Region.results, 
            filtered by the maximum price delta on each day according to the priceBands tuples. e.g. dpPres[datetime(2020,4,1)][(0,100)] contains the Region.results dataframe for
            the second quarter of 2020, filtered by days where the maximum energy price delta was between 0 and 100 $/MWh.
    
    Use this function to examine the long-term dynamics of the arbitrage market under different scenarios.
    
    Created on 8/7/2020 by Bennett Schneider
        
    """
    
    results = Region.RESULTS[metaVal][sMax]
    
     # shift back to get the end avg correct before resampling
    resampRes = results.shift(-1,freq=f"{freq}T").resample('d')
    
    # Create empty dataframe based on the resampled index
    dailyRes = pd.DataFrame(index=resampRes.mean().index)
    
    #Calculate daily profit and delta P
    dailyRes[r'Profit_$/MWh'] = resampRes['RRP_DailyProfit_$'].mean()
    dailyRes[r'Price_delta_$/MWh'] = resampRes[r'RRP'].max() - resampRes[r'RRP'].min()
    
    totalProfit = dailyRes[r'Profit_$/MWh'].sum()
    
    Region.dpPres = {} # add dpRes to the Region object
    
    
    if mode =='quarterly':
        quarters = {1:[1,2,3],2:[4,5,6],3:[7,8,9],4:[10,11,12]}
    else:
        quarters = {1:[np.nan]}
        
    for quarter,months in quarters.items():
        # dailyRes0 = dailyRes[dailyRes.index.month == month]
        dailyRes0 = dailyRes.loc[[i for i,m in zip(dailyRes.index,dailyRes.index.month) if m in months]]
        
        try:
            key = dt.datetime(dailyRes0.index[0].year,dailyRes0.index[0].month,1)
        
        except IndexError:
            dailyRes0 = dailyRes
        
        key = dt.datetime(dailyRes0.index[0].year,dailyRes0.index[0].month,1) # dt.datetime(int(year.split(',')[0]),7,1)
        
        Region.dpPres[key] = {}
                
        for dP in priceBands:
            # Slice the df to the daily price differential
            Region.dpPres[key][dP] = dailyRes0[(dailyRes0[r'Price_delta_$/MWh'] >= dP[0]) & (dailyRes0[r'Price_delta_$/MWh'] < dP[1])]

def intervalPriceBands(Region,priceBands,sMax,market='RRP',freq=5,intvl=30,mode='quarterly'):
    """
    For a given Region object, creates a nested dict of pandas dataframes containing the average of the results attribute in whatever interval desired, filtered by the average price over that interval, as
    indicated by the list of tuples in priceBands.
    
    Args:
        Region (Region): A Region object.
          - results (pandas DataFrame)
         
         priceBands (list of tups): Each tuple pair is a range of prices. The results attribute will be sorted accoring to these ranges.
         
         sMax (int): The maximum hours of storage for which you want to generate results.

         market (str. Default=RRP): Standard market price name by AEMO convention, e.g. 'RRP', 'RAISREGRRP','LOWER5MINRRP', etc.

         freq (int. Default=5): Frequency of Region.results in mins.

         intvl (int or str. Default=30): If a float, the frequency of the intervals you are considering in minutes. Otherwise, you can enter any valid resample str.
         
         mode (str. Default='quarterly'): Controls the periodicity of the output.
             quarterly -> dpPres keys are datetimes representing the start of a given quarter
             annual -> dpPres keys are the datetime representing the start of a given financial year
    
    Adds to Region as attribute:
        filtPriceRes (dict of dict of pandas DataFrames): Keys are datetimes, values are the tuples in priceBands, which are in turn keys for a pandas DataFrame which is Region.results, 
            filtered by the price in each interval according to the priceBands tuples. e.g. dpPres[datetime(2020,4,1)][(0,100)] contains the Region.results dataframe for
            the second quarter of 2020, filtered by intervals where the price was between 0 and 100 $/MWh.
    
    Use this function to examine the long-term dynamics of arbitrage and FCAS markets under different scenarios.
    
    Created on 8/7/2020 by Bennett Schneider
        
    """
    
    results = Region.results[sMax].copy()
    
    td = pd.Timedelta(minutes=freq)
    if type(intvl) == int:
        intvl_str = f"{intvl}T"
    else:
        intvl_str = intvl
        results.index = results.index - td

     # shift back to get the end avg correct before resampling
    resampRes = results.resample(intvl_str,loffset=pd.Timedelta(minutes=intvl))
    
    # Create empty dataframe based on the resampled index
    procRes = pd.DataFrame(index=resampRes.mean().index)
    
    # Calculate daily profit and delta P
    procRes[r'Profit_$/MWh'] = resampRes[f'{market}_Profit_$'].mean()
    procRes[market] = resampRes[market].mean()

    
    
    totalProfit = procRes[r'Profit_$/MWh'].sum()
    
    Region.filtPriceRes = {} # add dpRes to the Region object
    
    
    if mode =='quarterly':
        quarters = {1:[1,2,3],2:[4,5,6],3:[7,8,9],4:[10,11,12]}
    else:
        quarters = {1:[np.nan]}
        
    for quarter,months in quarters.items():
        # procRes0 = procRes[procRes.index.month == month]
        procRes0 = procRes.loc[[i for i,m in zip(procRes.index,procRes.index.month) if m in months]]
        
        try:
            key = dt.datetime(procRes0.index[0].year,procRes0.index[0].month,1)
        
        except IndexError:
            procRes0 = procRes
        
        key = dt.datetime(procRes0.index[0].year,procRes0.index[0].month,1) # dt.datetime(int(year.split(',')[0]),7,1)
        
        Region.filtPriceRes[key] = {}
                
        for dP in priceBands:
            # Slice the df to the daily price differential
            Region.filtPriceRes[key][dP] = procRes0[(procRes0[market] > dP[0]) & (procRes0[market] <= dP[1])]


def metaDataTrans(metaKey,metaVal,sep):
    """
    Takes a string of values separated by a given separator and a string containing the meanings of these values separated by the same separator.
    Returns a dictionary were the keys are each of the meanings and the values are the values, as paired based on the separator.
    """
    keys = metaKey.split(sep)
    vals = metaVal.split(sep)
    metaData = {key:val for key,val in zip(keys,vals)}
    return metaData

def createMetaVal(markets,freq,tFcst,tInt,eta,regDisFrac):
    """
    Custom function to return a specific string that is used as a results key. Functionalising for consistency
    """
    return f"{','.join(markets)}_{freq}_{tFcst}_{tInt}_{eta}_{regDisFrac}"

def createMetaVal2(shorthand,freq,tFcst,tInt,eta,regDisFrac,regDict_version):
    """
    Custom function to return a specific string that is used as a results key. Functionalising for consistency
    """
    return f"{shorthand}_{freq}_{tFcst}_{tInt}_{eta}_{regDisFrac}_{regDict_version}"

def stackMetaval(df,metaval):
    """
    Takes a pandas dataframe associated with a given metaval and assigns columns matching the names of each of the metaval elements and 
    filled with the value of that metaval element.
    """
    df = df.copy()
    mvList = metaval.split('_')
    metaDict = {
        'shorthand': 0,
        'freq': 1,
        'H': 2,
        't': 3,
        'eta': 4,
        'Fr': 5,
        'Fr_obj': 6
    }
    for key,val in metaDict.items():
        df[key] = mvList[val]

    return df

def create_tFcst(timeStamps,hr_frac,div=4,manual=None):
    """
    Will generate the hours over which your forecast will be visible to your optimiser. Will also return a string that
    can be used for your meta value key that is not dependent on the value of timeStamps or hr_frac.
    
    Args:
        timeStamps (int): Number of time-stamps that you are running through your optimiser.
        
        hr_frac (float): The fraction of an hour represented by each time-step that your optimiser is running over.
        
        div (int): Will divide the non-manual result by this number. This transformation will also be reflected in the string result.
        
    Returns:
        tFcst (int): If manual, just returns the value of manual. Otherwise, returns int(timeStamps*hr_frac/div)
        
        tFcst_str (str): If manual, just returns manual as a str. Otherwise, returns 'all/{div}'
    
    Use this function to improve the functionality of the meta-value key system which allows one to store multiple optimisation runs under different assumptions under a single unique key string.
    
    Created on 28/08/2020 by Bennett Schneider
    
    """
    
    if manual:
        tFsct = manual
        tFcst_str = str(manual)
    else:
        tFcst = int(timeStamps*hr_frac/div)
        tFcst_str = f'all/{int(div)}'
        
    return tFcst,tFcst_str

def create_tInt(timeStamps,hr_frac,div=4,manual=None):
    """
    Will generate the length the desired dispatch interval in minutes, i.e. the steps in which you want your optimiser to move forward in. Will also return a string that
    can be used for your meta value key that is not dependent on the value of timeStamps or hr_frac.
    
    Args:
        timeStamps (int): Number of time-stamps that you are running through your optimiser.
        
        hr_frac (float): The fraction of an hour represented by each time-step that your optimiser is running over.
        
        div (int): Will divide the non-manual result by this number. This transformation will also be reflected in the string result.
        
    Returns:
        tFcst (int): If manual, just returns the value of manual. Otherwise, returns int(timeStamps*hr_frac/div)
        
        tFcst_str (str): If manual, just returns manual as a str. Otherwise, returns 'all/{div}'
    
    Use this function to improve the functionality of the meta-value key system which allows one to store multiple optimisation runs under different assumptions under a single unique key string.
    
    Created on 28/08/2020 by Bennett Schneider
    
    """
    
    if manual:
        tInt = manual
        tInt_str = str(manual)
    else:
        tInt = int(timeStamps*hr_frac*60/div)
        tInt_str = f'all/{int(div)}'
        
    return tInt,tInt_str

def cycles(st,smax,window=False):
    """
    Takes a pandas series of the state of charge of a battery at a given time, t, as well as the storage capacity of the battery.
    Returns the cumulate number of cycles within the given period and the min and max state of charge.
    
    Args:
        st (pandas Series): Index should be a datetime index, but not required. Units of st and smax must match, but any unit will work. Hours is recommended.
        
        smax (float): The storage capacity of the battery. Hours is the recommended unit, but any unit will do, as long as smax and st are in the same unit
        
        window (bool or int. Default=False): If False, does nothing. Otherwise, this is the rolling average window to be applied to st to smooth out micro
            optimisations.
        
    Returns:
        num_cycles (float): Number of cycles across the time period.
        
        minSoC (float): Manimum SoC as a ratio of smax.
        
        maxSoC (float): Maximum SoC as a ratio of smax.
        
    Created on 30/9/2020 by Bennett Schneider
    
    """
    if window:
        dt_eff = st.rolling(window).mean().diff().abs() # absolute difference between successive states of charge in hours
    else:
        dt_eff = st.diff().abs() # absolute difference between successive states of charge in hours
    num_cycles = dt_eff.sum()/(2*smax) # add all the hours up over the given period and divide by twice the capacity of the battery (cycle is up and down)
    # min and max SoC over the given period
    minSoC = st.min()/smax
    maxSoC = st.max()/smax 
    return num_cycles,minSoC,maxSoC


def dictFilt(df,selectDict):
    """
    Filters a dataframe by a set of criteria stipulated in selectDict.
    
    Args:
        df (pandas DataFrame): Any pandas dataframe.
        
        selectDict (dict of iterables): Keys are column names for df. Values must be iterables. Will filter df based on
            whether the columns in the keys contain the values in the values of selectDict. If a value is entered in selectDict as None,
             it will not be used to filter df. If df does not contain a given key in selectDict in its columns, a warning is printed.
            
    Returns:
        df_filt (pandas DataFrame): df, but with rows filtered according to selectDict
    """
    df_filt = df.copy()
    for key,val in selectDict.items():
        if val:
            try:
                df_filt = df_filt[df_filt[key].isin(val)]
            except KeyError:
                print(f'WARNING: {key} not in df')
                pass
            except TypeError:
                print(f'key: {key}\nval:{val}')
                print('Try again!')
                sys.exit()
    
    return df_filt

# def thresh_smooth(rdata,freq,window,thresh,col,roll=False):
def thresh_smooth(data,freq,window=4,thresh=50,roll=False):
    """
    Implements a smoothing function that sets the value of data to the average value across a rolling 
    window if the differential across that rolling window is < thresh.
    
    RRP_6t' = {(1/h) ∑_t^(t+h)_t RRP_6t, |max⁡(RRP_6t)-min(RRP_6t)| < P
            RRP_6t,t∈h,|max⁡(RRP_6t)-min(RRP_6t)| >= P
    Args:
        # rdata (pandas Series or pandas DataFrame): Any pandas Series
        data (pandas DataFrame): Any pandas DataFrame
        
        freq (int): Frequency of the data in mins
        
        window (int): Number of intervals over which to implement rolling average.
        
        thresh (float): Threshold below which to flatten the data.
        
        col (str): Column of your pandas dataframe you want to apply this to, or name you 
            want to give to your series.
        
    Returns:
        threshData (pandas DataFrame): Smoothed data.
        
    Created on 13/11/2020 by Bennett Schneider
    
    """
    data = data.copy()
    
    DATA = []
    for col in data.columns:
        rdata = data[col]
        rdata.name = col
        # rname = col
        
        try:
            rdata = rdata.to_frame()
        except AttributeError:
            pass
        
        rdata.index = rdata.index - pd.Timedelta(minutes=freq)
        
        if roll:
            # rdata = rdata.rolling(window).apply(lambda dataSlice: thresh_flat(dataSlice,thresh,rname,roll=True))
            rdata = rdata.rolling(window).apply(lambda dataSlice: thresh_flat(dataSlice,thresh,roll=True),raw=True)
        else:
            rdata['group'] = [i//window for i in range(len(rdata))] # set up a grouping set
            # breakpoint()
            # rdata = rdata.groupby('group').apply(lambda dataSlice: thresh_flat(dataSlice,thresh,rname,roll=False))
            rdata = rdata.groupby('group')[col].apply(lambda dataSlice: thresh_flat(dataSlice,thresh,roll=False))
        # rdata = rdata[[rname]]

        DATA.append(rdata)

    # concatenate the series objects 
    data = pd.concat(DATA,axis=1)

    # shift the index
    data.index = rdata.index + pd.Timedelta(minutes=freq)

    # There will be some missing timestamps at the start due to the rolling average
    data.fillna(method='bfill',inplace=True)
    
    # add a new column to identify the new data based on the name of the functionhgo
    data['modFunc'] = thresh_smooth.__name__

    return data
    
    
# def thresh_flat(dataSlice,thresh,rname,roll):
def thresh_flat(dataSlice,thresh,roll):
    """
    The lambda function for thresh_smooth.
    """
    # print(dataSlice)
    
    if roll:
        dRange = abs(dataSlice.max() - dataSlice.min())
    else:
        # dRange = abs(dataSlice[group].max() - dataSlice[group].min())
        dRange = abs(dataSlice.max() - dataSlice.min())
    # print(dRange)
    # print(dataSlice.mean())
    if roll:
        if dRange < thresh:
            dataSlice = dataSlice.mean()
        else:
            dataSlice = dataSlice[-1]
    else:
        if dRange < thresh:
            # dataSlice[rname].loc[:] = dataSlice[rname].mean()
            dataSlice.loc[:] = dataSlice.mean()
    # print(dataSlice)
    return dataSlice

def stackData(data,index=None,values=None,resetindex=False):
    if resetindex:
        data = data.copy().reset_index()

    if index:
        pass
    elif values:
        index = [col for col in data.columns if col not in values]
    
    data = data.copy().set_index(index).stack().reset_index().rename({f'level_{len(index)}':'Variable',0:'Value'},axis=1)
    return data