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
path_base = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working"
t0 = dt.datetime(2020,1,1)
t1 = dt.datetime(2020,2,1)
region = 'NSW1'
myPath = os.path.join(path_base,r"results\bessOpt")
nem = go.NEM(myPath)
#%%
# Instantiate BESS objexct
bess1 = go.BESS(path,region,'bess1')
bess2 = go.BESS(path,region,'bess2')
bess3 = go.BESS(path,region,'bess3')

#%% Read in the price data
data_path = r"C:\Users\benne\OneDrive - Australian National University\Master of Energy Change\SCNC8021\packaging_working\RRP.csv"
RRP = pd.read_csv(data_path,index_col=0)
RRP.index = pd.to_datetime(RRP.index) # [dt.datetime.strptime(DT,'%Y-%m-%d %H:%M:%S') for DT in RRP.index]
RRP.index.name = 'Timestamp'
# load modified rrp into nem
nem.loadRaw(RRP, 5, 'Price')

#%% Run optimisation
m = Model(sense='MAX')
bess1.optDispatch(nem,m,t0,t1)
bess2.optDispatch(nem,m,t0,t1)
bess3.optDispatch(nem,m,t0,t1)
#%%
# bess1.revenue
# bess1.operations

# def plotResults(market='Energy'):
#     """
#     """

def rrpSchema(self,t0=None,t1=None):
    # Get the price
    
    rrp,rrp_mod = self.getRRP(nem,t0,t1)
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
    
def plottingSchema(self,t0=None,t1=None):   
    # https://stackoverflow.com/questions/62853539/plotly-how-to-plot-on-secondary-y-axis-with-plotly-express 
    # Process the rrp data into the desired schema
    RRP_plot = rrpSchema(self,t0=t0,t1=t1)
    
    # Construct a plot with plotly express based on just the price
    fig = px.line(RRP_plot,x='Timestamp',y='RRP',color='Direction',line_dash='Version',facet_row='Category').update_yaxes(matches=None)
    # plot(fig)

    
    # Get the operations table
    operations = self.operations.copy()
    
    # fig2 = self.energyDispatchPlot(show=False,t0=t0,t1=t1,kwargs={'figure':fig,'column':1,'row':1})
    # plot(fig2)
    
    # Get the revenue table
    revenue = self.revenue.copy()
    
    if t0 and t1:
        revenue = revenue[(revenue.index > t0) & (revenue.index <= t1)]
    
    # else:
    #     # Get the price data for the given interval
    #     t0 = revenue.index.min()
    #     t1 = revenue.index.max()
    
    # stackedOperations = self.stackOperations()
    # for col in ['Category','Direction','Market']:
    #     stackedOperations[col] = 'Energy'
        
    # stackedRevenue = self.stackRevenue()
    
    revenue = pd.merge(revenue.reset_index(),self.marketmeta,how='inner',on='Market')
    revenue.loc[revenue['Direction'] == 'LOWER','Dispatch_MW'] = -revenue['Dispatch_MW']
 
    fig2 = px.bar(revenue,x='Timestamp',y='Dispatch_MW',color='Direction',facet_row='Category',barmode='relative').update_yaxes(matches=None)
    
    # Just need to work out how to shift fig2 so it anchors to the secondary axis on fig 1. 
    # The below works for non-subplots, but not fir subplots
    fig2.update_traces(yaxis="y2")
    
    rows = 3
    cols = 1
    subfig = make_subplots(
        specs=rows*[cols*[{"secondary_y": True}]],
        rows=rows,
        cols=cols,
        # subplot_titles=subtitles
        )
    
    subfig.add_traces(fig.data + fig2.data)
    plot(subfig)

    # toPlot = pd.concat([stackedOperations,stackedRevenue])
    
plottingSchema(bess3,t0=t0,t1=t1)

#%% Plot
revenue = bess1.energyRevenuePlot(nem,show=True)
dispatch = bess1.energyDispatchPlot(show=True)
#%%
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as pgo
import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Graph(figure=dispatch,id='dispatch'),
    dcc.Graph(figure=revenue,id='revenue')
    ])


app.run_server(debug=True, use_reloader=False) 


# @app.callback(
#     dash.dependencies.Output('x-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#     country_name = hoverData['points'][0]['customdata']
#     dff = df[df['Country Name'] == country_name]
#     dff = dff[dff['Indicator Name'] == xaxis_column_name]
#     title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#     return create_time_series(dff, axis_type, title)


@app.callback(
    dash.dependencies.Output('revenue', 'figure'),
    [dash.dependencies.Input('dispatch', 'figure')])
def update_fig1(fig1):
    print(fig1)
    return None




