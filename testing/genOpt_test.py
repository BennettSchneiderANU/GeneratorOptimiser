import genOpt as go
import datetime as dt
#%% Initialise
t0 = dt.datetime(2020,1,1)
t1 = dt.datetime(2021,1,1)
path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
bess = go.BESS()