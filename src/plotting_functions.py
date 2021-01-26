import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.express as px
import plotly.colors
import os

def plotlyPivot(
    data,
    Line=[],
    Scatter=[],
    Bar=[],
    secondary_y=[],
    fill={},
    stackgroup={},
    stacked=False,
    relative=False,
    logy=False,
    log2y=False,
    labels={},
    ylabel='',
    y2label='',
    xlabel='',
    title='',
    exportHTML=False,
    exportPNG=False,
    width=1800,
    height=1050,
    show=True,
    ms=5,
    lw=2,
    mode='lines',
    colorList=plotly.colors.qualitative.Pastel,
    fontsize=16,
    symbol=0,
    opacity=0.9,
    legendDict=None,
    xlim=None,
    ylim=None,
    y2lim=None,
    legend_traceorder='normal'
    ):
    """
    A customisable multi-style plotly plotter. Creates combined Line, Scatter and Bar charts with secondary axis options out of
    a single pandas DataFrame. If no columns are entered into Line, Scatter, Bar, then all are plotted as Line.
    
    Args:
        data (pandas DataFrame): Any pandas DataFrame.
        
        Line (list of strs. Default=[]): List of column names in data you want to plot on a line plot.
        
        Scatter (list of strs. Default=[]): List of column names in data you want to plot on a Scatter plot (passing the same str to 'fill' with
            the appropriate value can turn this into a customisable area plot).
        
        Bar (list of strs. Default=[]): List of column names in data you want to plot on a Bar plot.
        
        secondary_y (list of strs. Default=[]): List of column names in data that you want to plot on a secondary y axis.
        
        fill (dict. Default={}): Dict with keys as column names that you want to turn into area plots. These column names must also appear in Scatter.
            Acceptable values are:
              - 'tozeroy'
              - 'tonexty'
        
        stackgroup (dict of lists): Key is the name of your stack group. List is the column names in that stack. Use this to make stacked area charts. Only works for scatter.

        stacked (bool. Default=False): Only used for if entries in Bar. If True, makes bar chart stacked.

        relative (bool. Default=False): Like stacked, but puts negative values below the axis.

        logy (bool. Default=False): If False, use linear scale on primary y axis. If True, uses log scale.

        log2y (bool. Default=False): As for logy, but pertains to the secondary y axis.
        
        labels (dict. Default={}): Keys are the original column names. Values are the corresponding strings you want to use in the legend.
            If you don't enter a value here, the column name is used as-is.
        
        ylabel (str. Default=''): Label for primary y axis.
        
        y2label (str. Default=''): Label for secondary y axis.
        
        xlabel (str. Default=''): Label for primary x axis.
        
        title (str. Default=''): Title for the plot.
        
        exportHTML (str or bool. Default=False): If False, doesn't do anything. If True, saves to this location as an HTML file, and 
            also shows the plot.
        
        show (bool. Default=True): If True, shows the plot in browser.

        ms (int. Default=5): Marker size (only for scatter plots).

        lw (int. Default=2): Line widtth (only for scatter plots).

        mode (str. Default='lines'): The 'mode' property is a flaglist and may be specified as a string containing:
            - Any combination of ['lines', 'markers', 'text'] joined with '+' characters (e.g. 'lines+markers') 
              OR exactly one of ['none'] (e.g. 'none') 
              Only affects plot if 'Line' is used.
        
        colorList(list of strs. Default=plotly.colors.qualitative.Pastel): List of colours. See here for options: https://plotly.com/python/discrete-color/.
            Can also enter as a dict of lists with keys 'Scatter','Bar','Line' to assign different color lists to different trace types.

        fontsize (int. Default=20): Fontsize of plot.

        symbol (str): Any valid marker str from marker/symbol: https://plotly.com/python/reference/

        opacity (float.Default=0.9): Number between 0 and 1 where 0 is transparent, 1 is opaque. 
        
    
    Returns:
        fig (plotly figure): Can be displayed in browser by calling plot(fig,auto_open=True)
              
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if len(Line) == 0 and len(Bar) == 0 and len(Scatter) == 0:
        Line = list(data.columns)

    if type(colorList) == list:
        colorList = {'Scatter': colorList,'Line': colorList,'Bar': colorList}

    count = -1
    for col in Scatter:
        count = np.mod(count + 1,len(colorList['Scatter']))
        try:
            name = labels[col]
        except KeyError:
            name = col
            
        if col in secondary_y:
            sy = True
        else:
            sy = False
        
        try:
            f = fill[col]
        except KeyError:
            f = None
        
        try:
            stack = stackgroup[col]
        except:
            stack = None
        
        fillcolor = f"rgba{colorList['Scatter'][count][3:-1]}, {opacity})"
        # breakpoint()
        fig.add_trace(
            go.Scatter(
                x = data.index,
                y = data[col],
                fill = f,
                stackgroup = stack,
                name = name,
                fillcolor=fillcolor,
                marker = dict(
                    color=colorList['Scatter'][count],
                    size=ms,
                    line=dict(width=2,color='DarkSlateGrey'),
                    opacity=opacity
                ),
                line=dict(
                    width=0
                )
                ),
            secondary_y = sy
            )
    
    count = -1
    for col in Line:
        count = np.mod(count + 1,len(colorList['Line']))

        try:
            name = labels[col]
        except KeyError:
            name = col
            
        if col in secondary_y:
            sy = True
        else:
            sy = False

        fig.add_trace(
            go.Line(
                x = data.index,
                y = data[col],
                name = name,
                mode=mode,
                marker = dict(
                    color=colorList['Line'][count],
                    size=ms,
                    symbol=symbol,
                    line=dict(width=2,color='DarkSlateGrey')
                ),
                line = dict(
                    color=colorList['Line'][count],
                    width=lw,
                )
                ),
            secondary_y = sy
            )
        
        # fig.update_traces(
        #     marker=dict(
        #         size=ms,
        #         line=dict(width=2,color='DarkSlateGrey')
        #         )
        # )
    
    count = -1
    for col in Bar:
        count = np.mod(count + 1,len(colorList['Bar']))
        try:
            name = labels[col]
        except KeyError:
            name = col
            
        if col in secondary_y:
            sy = True
        else:
            sy = False
            
        fig.add_trace(
            go.Bar(
                x = data.index,
                y = data[col],
                name = name,
                marker = dict(
                    color=colorList['Bar'][count]
                )
                ),
            secondary_y = sy
            )
        
        if stacked:
            fig.update_layout(barmode='stack')
        elif relative:
            fig.update_layout(barmode='relative')
    
    
    # fig.update_yaxes(title_text=ylabel)
    # fig.update_yaxes(title_text=y2label,secondary_y=True)
    # fig.update_xaxes(title_text=xlabel)
    
    if logy:
        type_y = 'log'
    else:
        type_y ='linear'
    
    if log2y:
        type_2y = 'log'
    else:
        type_2y ='linear'

    fig.update_layout(
        legend=legendDict,
        title=title,
        yaxis=dict(
            title=ylabel,
            type=type_y,
            range=ylim
        ),
        yaxis2=dict(
            title=y2label,
            type=type_2y,
            range=y2lim
        ),
        xaxis=dict(
            title=xlabel,
            range=xlim
        ),
        font={
            'size':fontsize
        },
        legend_traceorder=legend_traceorder
        )

    if exportHTML:
        plot(fig,filename=exportHTML)

    if exportPNG:
        fig.write_image(exportPNG,width=width,height=height)

    if show and not exportHTML:
        plot(fig,auto_open=True)
        
    return fig

def plotlyStack():
    pass