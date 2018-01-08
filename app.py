import os

server = app.server
server.secret_key = os.environ.get('SECRET_KEY', 'fghjkl')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.plotly as py
from operator import add
import dash; from dash.dependencies import Input, Output
import dash_core_components as dcc; import dash_html_components as html
import dash_html_components as html
import plotly.graph_objs as go


#data preparation 




ufos=pd.read_csv('complete.csv',error_bad_lines=False,low_memory=False)
ufos['year'] =  ufos.datetime.apply(lambda x: x.split('/')[2].split(' ')[0])
ufos['month'] = ufos.datetime.apply(lambda x: x.split('/')[0].split(' ')[0]).astype('int')
ufos.year = ufos.year.astype('int')
scl = [[0.0, 'rgb(242,240,247)'],[1.0, 'rgb(155,0,0)']]
years = pd.Series(ufos['year'].unique())[pd.Series(ufos['year'].unique()).astype('str').str.slice(3,4)=='0']
levels = ufos.groupby('shape').size().sort_values(ascending = False).index.get_values().tolist()

#data for line plot
ldata=[]
for i in range(len(ufos['shape'].unique().tolist())):
    dat=ufos[ufos['shape']==ufos['shape'].unique().tolist()[i]].groupby('year').datetime.count().reset_index()
    dat.rename(columns={'year': 'year', 'datetime': 'count'}, inplace=True)
    trace = go.Scatter(
    name = ufos['shape'].unique().tolist()[i],
    x = dat['year'],
    y = dat['count'],
    mode = 'lines')
    ldata.append(trace)
llayout = dict(showlegend = False,
hovermode= 'closest',plot_bgcolor = "#0D0D0D",
yaxis =dict(
type='ln',title = 'n',
autorange=True))

#data for choroplex plot
ufos[ufos.country=='us'].groupby('state').datetime.count()
counts=ufos[ufos.country=='us'].groupby('state').datetime.count()
counts=counts[~np.isnan(counts)]

counts=counts.to_frame().reset_index()
counts.rename(columns={'state': 'state', 'datetime': 'count'}, inplace=True)
counts['state'] = counts.state.str.upper()



app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='UFO Sightings', style={'text-align': 'center','color': '#000444', 'fontSize': 24, 'font-weight': 'bold'}),
    html.H2(children='Reports of unidentified flying object reports in the last century', style={'text-align': 'center','color': '#000444', 'fontSize': 18, 'font-weight': 'bold'}),
    dcc.Markdown('''
UFO Sightings [dataset](https://www.kaggle.com/NUFORC/ufo-sightings) was dowloaded from Kaggle.

##### Main questions we wanted to answer:
1) What is distribution of UFO reports among different states?
2) How does the number of UFO occurencies change in time?
3) What is the most popular alien aircraft?
4) Do UFOs tend to show at a particular daytime?
5) Which percentage of all UFOs report could be explained by nearby airports?

____


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.plotly as py
from operator import add
from datetime import datetime


Now we would download UFOs dataset:

```python
%%capture
ufos=pd.read_csv('complete.csv',error_bad_lines=False,low_memory=False)
```
____
###### 1) What is distribution of UFO reports among different states?

```python
ufos[ufos.country=='us'].groupby('state').datetime.count()
counts=ufos[ufos.country=='us'].groupby('state').datetime.count()
counts=counts[~np.isnan(counts)]
counts=counts.to_frame().reset_index()
counts.rename(columns={'state': 'state', 'datetime': 'count'}, inplace=True)
counts['state'] = counts.state.str.upper()
counts.sort_values(by='count',ascending = False).iloc[0]
```

>state      CA
>count    9575
>Name: 4, dtype: object    
____   

    ''' ),
    dcc.Graph(id = 'by_states',
             figure= dict( data=[ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = counts['state'],
        z = counts['count'].astype(str),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(0,0,0)',
                width = 1.5
            ) ),
        colorbar = dict(
            title = "UFOs")
        ) ], 
        layout=dict(
        width = 1200,
        height = 800,
        title = ' ',
        geo = dict(
            scope='usa',width = 1000,
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             ) )),
    dcc.Markdown('''
    
California is the most popular state by aliens.

___
    
###### 2) How does the number of UFO occurencies change in time?
    
```python
#counts by years
ufos['year'] =  ufos.datetime.apply(lambda x: x.split('/')[2].split(' ')[0])
yrs = ufos.groupby('year').datetime.count()
yrs.to_frame().reset_index()
yrs.rename(columns={'year': 'year', '0': 'count'}, inplace=True)
ufos['year'] =  ufos.datetime.apply(lambda x: x.split('/')[2].split(' ')[0])
yrs = ufos.groupby('year').datetime.count().to_frame().reset_index()
yrs.rename(columns={'year': 'year', 'datetime': 'count'}, inplace=True)
yrs.rename(columns={'year': 'year', '0': 'count'}, inplace=True)
```

Number of UFO reports tends to increase dramatically during the end of XX century.

___
    
###### 3) What is the most popular alien aircraft?

```python
ufos.groupby('shape').datetime.count().sort_values(ascending=False).head(1)
```
>shape
>light       17872
>triangle     8489
>circle       8453
>Name: datetime, dtype: int64

The most popular aircraft appearance is light followed by triange and circle

___

###### 4) Do UFOs tend to show at a particular daytime?

There are hours in 24:00 format which must be changed into 00
```python
mask = []
for value in ufos.datetime:
    res = value.split('/')[2].split(' ')[1]=='24:00'
    mask.append(res)
ufos.loc[mask,'datetime']=ufos.loc[mask,'datetime'].apply(lambda x: x.replace('24','00'))
mask = ufos.datetime.apply(lambda x: x.split('/')[1])!='00'
sum(ufos.datetime.apply(lambda x: x.split('/')[1])=='00') #there are 45 dates in format ??/00/?? so we won't include them into analysis
ufos_times = ufos.loc[mask]
stamps =  ufos_times.datetime.apply(lambda x:datetime.strptime(x, '%m/%d/%Y %H:%M').strftime('%H%M') )
pd.options.mode.chained_assignment = None  
ufos_times.loc[:, 'stamps'] = stamps
```

We assume next dividing of the day (ignoring seasonal differences):
6:00-18:00 day
18:00 - 6:00 night

```python
ufos_times.stamps= ufos_times.stamps.astype('int')
ufos_times['daytime'] = ['?']*ufos_times.shape[0]
mask = (ufos_times.stamps>600)&(ufos_times.stamps<1800)
ufos_times.loc[mask,'daytime'] = 'day'
ufos_times.loc[~mask,'daytime'] = 'night'
ufos_times.daytime.value_counts()
ufos_times.daytime.value_counts()[0]/ufos_times.daytime.value_counts()[1]
```
>4.0807681284035544

>As we can see, aliens are noticed mainly during nighttime

___

###### 5) Which percentage of all UFOs report could be explained by nearby airports?
Higher number of occurencies near airports can be also explained by higher interest in human technology among aliens.

We will compute the fraction of all UFO observations which have an airport located nearby (<10km).

```python
#Downloading airport location data
airports=pd.read_csv('airports1.csv',sep=';')
airports.country = airports.country.str.lower()
airports.country=airports['country'].apply(lambda x: x.replace(' ','_') )
us_airports = airports[airports.country=='united_states']
```
____

Now we will compute the distance between UFO observation and all US airports using Haversine formula:

![alt text](https://user-images.githubusercontent.com/2789198/27240432-e67a0cf0-52d4-11e7-9acb-b935e1a84f47.png)



```python
import math
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d
```



In the next code airports located <10 km are selected
```python
airport_codes = us_airports.ICAO.unique()
#computing if there is airport (distance < 10km ) near  each UFO occurence:
airport_nearby = [0]*ufos.shape[0]
for i in range(ufos.shape[0]):
    occ = (float(ufos.iloc[i,].latitude),float(ufos.iloc[i,].longitude)) # particular UFO occurence
    for j in range(len(airport_codes)):
        dist = distance(occ,(us_airports.loc[us_airports.ICAO == airport_codes[j],].Latitude,us_airports.loc[us_airports.ICAO == airport_codes[j],].Longitude))
        if dist<10:
            airport_nearby[i]=1
            break
        elif i == 1434:
            airport_nearby[i]=0

```
It's not possible to compute 80000*1400 distances on my comp, so we'll use the sample:
```python
airport_nearby = [0]*10000
for i in range(10000):
    occ = (float(ufos.iloc[i,].latitude),float(ufos.iloc[i,].longitude)) # particular UFO occurence
    for j in range(len(airport_codes)):
        dist = distance(occ,(us_airports.loc[us_airports.ICAO == airport_codes[j],].Latitude,us_airports.loc[us_airports.ICAO == airport_codes[j],].Longitude))
        if dist<10:
            airport_nearby[i]=1
            break
        if j == 1434:
            airport_nearby[i]=0
sum(airport_nearby)/len(airport_nearby)
```
>0.345

After computing for 10 000 observations (~3h hours) it's pretty clear that the fraction of UFO observations with nearby airports converges to ~35%
There is also possibility of military airplane misinterpreted as UFOs, but I failed to find any dataset contatining locations of US military bases (what a suprise!).
____

    '''),
html.Div(id = 'year', style={'color': '#000444', 'fontSize': 20, 'font-weight': 'bold'}),
  dcc.Graph(id='line-graph'),
    html.Div([
        html.Div([
            dcc.Graph(id='shapes-graph')
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='chloroplex-graph')
        ], className="six columns"),
    ], className="row"),
    dcc.Slider(
        id='year-slider',
        min=ufos.year.min(),
        max=ufos.year.max(),
        value=ufos.year.min(),
        step=None,
        updatemode = 'mouseup',
        marks={str(year): '' for year in ufos['year'].unique()}
    )
])

@app.callback(
    Output(component_id='year', component_property='children'),
    [Input(component_id='year-slider', component_property='value')]
)
def update_output_div(selected_year):
    return 'United States of America,{}'.format(selected_year)


#Line chart
@app.callback(
    dash.dependencies.Output('line-graph', 'figure'),
    [dash.dependencies.Input('year-slider', 'value')])


def update_line_chart(selected_year):
    ldata=[]
    df = ufos[ufos['year']==selected_year]
    for i in range(len(ufos['shape'].unique().tolist())):
        dat=df[df['shape']==ufos['shape'].unique().tolist()[i]].groupby('month').datetime.count().reset_index()
        dat.rename(columns={'month': 'month', 'datetime': 'count'}, inplace=True)
        trace = go.Bar(
        marker = dict(line = dict(
                color='rgb(0,0,0)',
                width=1.5)),
        width = 0.3,
        name = ufos['shape'].unique().tolist()[i],
        x = dat['month'],
        y = dat['count'])
        ldata.append(trace)
    dat = df.groupby('month').datetime.count().reset_index()
    dat.rename(columns={'month': 'month', 'datetime': 'count'}, inplace=True)
    trace = go.Scatter(
    name = ufos['shape'].unique().tolist()[i],
    x = dat['month'],
    y = dat['count'],
    mode = 'lines')
    ldata.append(trace)

    llayout = go.Layout(
    barmode = 'stack',
    showlegend = False,
    hovermode= 'closest',
    plot_bgcolor = "#0D0D0D",
    xaxis = dict(title = 'month'),
    yaxis =dict(
        type='ln',title = 'n',
        autorange=True)
    )
    return {
        'data': ldata,
        'layout': llayout
    }



#Chloroplex chart
@app.callback(
    dash.dependencies.Output('chloroplex-graph', 'figure'),
    [dash.dependencies.Input('year-slider', 'value')])


def update_figure(selected_year):
    df = ufos[ufos['year']==selected_year]
    data = [ dict(
        type = 'scattergeo',
        showlegend = False,
        locationmode = 'USA-states',
        lon = df['longitude'],
        lat = df['latitude'],
        text = df['state'],
        mode = 'markers',
        hoverinfo='none',
        marker = dict(
            size = 2,
            color='yellow',
            opacity = 0.99,
            reversescale = False,
            autocolorscale = False,
            hoverinfo='none',
            line = dict(
                width=0.02,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            colorbar=False
            )
        )]
    layout = dict(
        margin=go.Margin(
        l=0,
        r=0,
        b=50,
        t=0,
        pad=0),
        colorbar = True,
        showlegend = False,
        plot_bgcolor = '#0D0D0D',
        geo = dict(
            bgcolor = "#0D0D0D",
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            showlegend = False,
            showscale= False,
            landcolor = "rgb(0, 0, 0)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        )
    )
    return {
        'data': data,
        'layout': layout
    }

#Histogram plot
@app.callback(
    Output(component_id='shapes-graph', component_property='figure'),
    [Input(component_id='year-slider', component_property='value')]
)

def update_figure(selected_year):
    df = ufos[(ufos['year']==int(selected_year))&(ufos['country']=='us')]
    #add null-shape observations to every one-year df:
    for i in range(len(ufos['shape'].unique())):
        if not ufos['shape'].unique()[i] in df['shape'].unique():
            null = pd.DataFrame({'shape':[ufos['shape'].unique()[i]],'year':['1945']},columns = ['year','shape'])
            df = df.append(null)
        else:
            notnull = pd.DataFrame({'shape':[ufos['shape'].unique()[i]],'year':['1945']},columns = ['year','shape'])
            df = df.append(notnull)
    df['shape'] = df['shape'].astype('category')
    df['shape'] =  df['shape'].cat.reorder_categories(levels)
    counts = df[['shape','year']].groupby('shape').agg('count').reset_index()
    counts.columns = ['shape','number']

    data = [go.Bar(
    x =counts['shape'],
    y =counts['number']-1,
    text = counts['number']-1,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
        color='rgb(8,48,107)',
        width=1.5)),
    hoverinfo='none'
    )]

    margin=go.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    )
    
    figure = {
        'data': data,
        'layout': {
            'showlegend': False,
            'margin':{'l':'40','r':'0','b':'70','t':'20','pad':''},
            'yaxis': {'title': 'n of occurencies','range':[0,100 if counts['number'].max()<100 else counts['number'].max()]},
            'plot_bgcolor':"#0D0D0D"
        }
        }
    return figure



app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server()
