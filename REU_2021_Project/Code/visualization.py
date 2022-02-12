#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:22:06 2021

@author: r21sscott

code for visualizing my data
many different ones in here
currently holds visualizations for labeled data:
    -how raw corpus was divided
    -different ways for distributin of themes in labeled data (waffle, sunburst, stacked bar graph)
    -training/test data results
    -models as an analytic tool results

"""
import csv
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.io as pio
from pywaffle import Waffle
import matplotlib.pyplot as plt
from nxviz.plots import CircosPlot
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import calendar
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, LeakyReLU, Bidirectional
pio.renderers.default = "firefox"

def main():
    print('working with waff data')
    #labelPie()
    #sunTwo()
    #addToVisData()
    #waffleSub()
    #waffleWhole()
    #bar()
    #paperPie()
    #themePie()
    classAccBar()
    #nnAccBar()
    #df_waff, yc, xc = waffData()
    #waff(df_waff, yc, xc)
    #compWaff(df_waff, yc, xc)

    
        
    
#add paper number column to visualization dataset
def addToVisData():
    #scrap2 = copy of an essays section in the training data (all rows from Federalist No. to PUBLIUS) just manually copy and pasted
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/scrap2.txt') 
    df1 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    df.insert(0,'Paper','57') #change to match paper you're working with
    with open('/mnt/linuxlab/home/r21sscott/REU_Project/visData1', 'a') as file:
            writer = csv.writer(file)
            #writer.writerow(['Theme', 'Document']) #uncomment if new file
            for x in range(len(df.index)):
                writer.writerow([df.Paper.iloc[x], df.Theme.iloc[x],df.Document.iloc[x]])
            

#plotly stacked bar graph to show distribution of themes in each essay 
def bar():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    papers = df.Paper.unique()
    labels = df.Theme.unique()
    rows = [] 
    temp = []

    #creates a list of the count of each theme for each paper
    for x in range(len(labels)):
        for y in range(len(papers)):
            p = df[df['Paper'] ==papers[y]]
            c = (p.Theme == labels[x]).sum()
            temp.append(c)
        rows.append(temp)
        temp = []
    
    #creates a stacked bar graph with each paper having a bar divided by theme
    fig = go.Figure(data=[
        go.Bar(name='none', x=papers, y=rows[0], marker_color='lightslategrey'),
        go.Bar(name='strong federal government', x=papers, y=rows[1], marker_color='navy'),
        go.Bar(name='legislative', x=papers, y=rows[2], marker_color='darkgreen'),
        go.Bar(name='nationalist outlook', x=papers, y=rows[3], marker_color='darkseagreen'),
        go.Bar(name='weakness of articles of Confederation', x=papers, y=rows[4], marker_color='gold'),
        go.Bar(name='judicial', x=papers, y=rows[5]),
        go.Bar(name='federalism', x=papers, y=rows[6]),
        go.Bar(name='natural rights', x=papers, y=rows[7]),
        go.Bar(name='union', x=papers, y=rows[8]),
        go.Bar(name='human nature', x=papers, y=rows[9], marker_color='purple'),
        go.Bar(name='interest-group theory', x=papers, y=rows[10]),
        go.Bar(name='republican principles', x=papers, y=rows[11]),
        go.Bar(name='checks and balances', x=papers, y=rows[12], marker_color='firebrick'),
        go.Bar(name='executive', x=papers, y=rows[13]),
        go.Bar(name=labels[14], x=papers, y=rows[14]),
    ])
    fig.update_xaxes(type='category')
    fig.update_xaxes(categoryorder='category ascending')
    fig.update_layout(title='Theme Breakdown in Labled Data',
                      title_font_size=20,
                      xaxis_title= 'Federalist Paper No.',
                      yaxis_title='Frequency of Theme',
                      barmode='stack')
    fig.update_xaxes(title_font={"size": 16},
                     tickfont=dict(size=12))
    fig.update_yaxes(title_font={"size": 16},
                     tickfont=dict(size=12))
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))
    fig.show()


# creates a waffle chart of the whole distribution of the training data
def waffleWhole():
    df_raw = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    df = df_raw.groupby('Theme').size().reset_index(name='counts')
    n_cats = df.shape[0]
    '''colors = [plt.cm.inferno_r(i/float(n_cats)) for ifig.update_layout(
    grid= dict(columns=2, rows=1),
    margin = dict(t=0, l=0, r=0, b=0)
    ) in range(n_cats)]'''
    fig = plt.figure(
    FigureClass=Waffle,
    plots={
        '111':{
            'values': df['counts'],
            'labels': ["{0} ({1})".format(n[0], n[1]) for n in df[['Theme', 'counts']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8},
            'title': {'label': 'Distribution of Themes in Labeled Data', 'loc': 'center', 'fontsize':18}
            },
        },
    rows = 20,
    columns = 20, 
    cmap_name="tab20",
    #figsize=(10,10)
    )
    
#creates waffle charts for each labeled Federalist Paper
#can only do 2 at a time or else it gets weird/crowded/messy
def waffleSub():
    #prepare data
    df_raw = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    papers = df_raw.Paper.unique()
    labels = df_raw.Theme.unique()
    cols = []
    temp = []
    #r = df_raw[df_raw['Paper'] ==papers[0]]
    #t = (x.Theme == 'none').sum()

    for x in range(len(papers)):
        for y in range(len(labels)):
            p = df_raw[df_raw['Paper'] ==papers[x]]
            c = (p.Theme == labels[y]).sum()
            temp.append(c)
        cols.append(temp)
        temp = []
    #print(cols)
       
    data = pd.DataFrame(
            {
                'labels':labels,
                '42': cols[0],
                '62': cols[1],
                '70': cols[2],
                '23': cols[3],
                '5': cols[4],
                '78': cols[5],
                '33': cols[6],
                '16': cols[7],
                '83': cols[8],
                '61': cols[9],
                '50': cols[10],
                '11': cols[11],
                '39': cols[12],
                '48': cols[13],
                '27': cols[14],
                '57': cols[15],
            },
        ).set_index('labels')
    #uncomment for the paper you want
    '''fig = plt.figure(
            FigureClass=Waffle,
            plots={
                '311': {
                    'values': data['42'],
                    'labels': [f"{k} ({v})" for k, v in data['42'].items()],
                    'legend': {
                        'loc': 'upper left',
                        'bbox_to_anchor': (1.5, 1.5),
                        'fontsize': 6
                    },
                    'title': {
                        'label': 'Federalist Paper No. 42',
                        'loc': 'left'
                    }
                },
                '312': {
                    'values': data['62'],
                    'labels': [f"{k} ({v})" for k, v in data['62'].items()],
                    'legend': {
                        'loc': 'upper left',
                        'bbox_to_anchor': (1.4, 1),
                        'fontsize': 6
                    },
                    'title': {
                        'label': 'Federalist Paper No. 62',
                        'loc': 'left'
                        }
                },
            },
            rows=5,
            cmap_name='tab20',  # shared parameter among subplots
            figsize=(9, 5)  # figsize is a parameter of plt.figure
        )
    fig2 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['70'],
                'labels': [f"{k} ({v})" for k, v in data['70'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1.5),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 70',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['23'],
                'labels': [f"{k} ({v})" for k, v in data['23'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.5, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 23',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 5)  # figsize is a parameter of plt.figure
    )
    fig3 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['5'],
                'labels': [f"{k} ({v})" for k, v in data['5'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (2, 1.2),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 5',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['78'],
                'labels': [f"{k} ({v})" for k, v in data['78'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 78',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 5)  # figsize is a parameter of plt.figure
    )
    fig4 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['33'],
                'labels': [f"{k} ({v})" for k, v in data['33'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1.2),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 33',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['16'],
                'labels': [f"{k} ({v})" for k, v in data['16'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 16',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 5)  # figsize is a parameter of plt.figure
    )
    fig5 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['83'],
                'labels': [f"{k} ({v})" for k, v in data['83'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 83',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['61'],
                'labels': [f"{k} ({v})" for k, v in data['61'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 61',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 5)  # figsize is a parameter of plt.figure
    )
    fig6 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['50'],
                'labels': [f"{k} ({v})" for k, v in data['50'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1.2),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 50',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['11'],
                'labels': [f"{k} ({v})" for k, v in data['11'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 11',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 6)  # figsize is a parameter of plt.figure
    )
    fig7 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['39'],
                'labels': [f"{k} ({v})" for k, v in data['39'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1.2),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 39',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['48'],
                'labels': [f"{k} ({v})" for k, v in data['48'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 48',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 6)  # figsize is a parameter of plt.figure
    )'''
    fig8 = plt.figure(
        FigureClass=Waffle,
        plots={
            '311': {
                'values': data['27'],
                'labels': [f"{k} ({v})" for k, v in data['27'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1.2),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 27',
                    'loc': 'left'
                }
            },
            '312': {
                'values': data['57'],
                'labels': [f"{k} ({v})" for k, v in data['57'].items()],
                'legend': {
                    'loc': 'upper left',
                    'bbox_to_anchor': (1.2, 1),
                    'fontsize': 6
                },
                'title': {
                    'label': 'Federalist Paper No. 57',
                    'loc': 'left'
                }
            },
        },
        rows=5,
        cmap_name='tab20',  # shared parameter among subplots
        figsize=(9, 6)  # figsize is a parameter of plt.figure
    )
    
    


#make csv file for sunburst plot
def addSunData():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1') #filepath depends on user
    with open('/mnt/linuxlab/home/r21sscott/REU_Project/sunburstData', 'a') as file:
            writer = csv.writer(file)
            #writer.writerow(['Theme', 'Document'])
            for x in range(len(df.index)):
                writer.writerow([str(df.Theme.iloc[x])+'-'+str(df.Document.iloc[x]),
                                 str(df.Paper.iloc[x])+'-'+str(df.Theme.iloc[x]),
                                 df.Document.iloc[x],'n/a'])   
                file.close()


#sunburst pie chart -doesn't currently work but would be preferred
def sunOne():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/sunburstData')
    df1 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    fig = go.Figure()
    fig.add_trace(go.Sunburst(
        ids=df.ids,
        labels=df.labels,
        parents=df.parents,
        domain=dict(column=0)
    ))
    #fig.update_traces(values=df.values, selector=dict(type='sunburst'))
    fig.show()

#easy sunburst-works
#good for pure visualization; hoverdata could be improved
def sunTwo():
    df1 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    fig = px.sunburst(df1,
                      path=['Paper', 'Theme', 'Document'],
                      title='Distribution of Themes in Labeled Data',
                      width=1000, height=1000)
    pio.show(fig)

#basic pie chart to show labeled vs. unlabeled ratio
def labelPie():
    labels = 'Labeled Papers', 'Unlabeled Papers'
    sizes = [16,69]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title('Use of The Federalist Papers')
    plt.show()

#pie chart for breakdown of essays in training data
def paperPie():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    papers = df.Paper.unique()
    values = []
    for x in range(len(papers)):
        c = (df.Paper == papers[x]).sum()
        values.append(c)
    fig = go.Figure(data=[go.Pie(labels=papers, values=values)])
    fig.update_layout(title='Distribution of Essays in Data')
    fig.update_traces(textinfo='value+percent')
    fig.show()
    
#pie chart for breakdown of themes in training data
def themePie():
    df = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/visData1')
    labels = df.Theme.unique()
    values = []
    for x in range(len(labels)):
        c = (df.Theme == labels[x]).sum()
        values.append(c)
    '''fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title='Distribution of Themes in Data')
    fig.update_traces(textinfo='value+percent')
    fig.show()'''
    fig1, ax1 = plt.subplots()
    ax1.pie(values, shadow=True, startangle=90)
    plt.legend()
    ax1.axis('equal')
    plt.title('Distribution of Themes in Training Data')
    plt.show()
    
    
#bar graph to show different accuracy scores for classifiers
#chose f1-scores due to the possibly overfitting of cross-val
def classAccBar():
    labels = ['LinearSVC', 'LogRegression', 'MultiNB', 'RandForest']
    ogAcc = [0.64, 0.59, 0.58, 0.6] 
    osAcc = [0.54, 0.56, 0.35, 0.64]
    smAcc = [0.51, 0.56, 0.36, 0.65]
    
    '''labels = ['Voting', 'Bagging', 'XGB', 'Gradient']
    ogAcc = [0.59, 0.6, 0.62, 0.59]
    osAcc = [0.57, 0.55, 0.62, 0.55]
    smAcc = [0.56, 0.6, 0.63, 0.62]'''
    
    x = np.arange(len(labels)) #label locations
    width = 0.2 #width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, ogAcc, width, label='Orginal dataset')
    rects2 = ax.bar(x + width, osAcc, width, label = 'Random Oversampling')
    rects3 = ax.bar(x + width*2, smAcc, width, label = 'SMOTE Oversampling')
    
    #adding text and such
    ax.set_ylabel('F1 Accuracy Scores')
    ax.set_xlabel('Type of Classifier')
    ax.set_title('Accuracy Scores of Test Data for Pre-exsisting Classifiers')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    
    #bar chart for ensemble classifiers
    '''labels = ['Voting', 'Bagging', 'XGB', 'Gradient']
    ogAcc = [0.59, 0.6, 0.62, 0.59]
    osAcc = [0.57, 0.55, 0.62, 0.55]
    smAcc = [0.56, 0.6, 0.63, 0.62]
    
    x = np.arange(len(labels)) #label locations
    width = 0.2 #width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, ogAcc, width, label='Orginal dataset')
    rects2 = ax.bar(x + width, osAcc, width, label = 'Random Oversampling')
    rects3 = ax.bar(x + width*2, smAcc, width, label = 'SMOTE Oversampling')
    
    #adding text and such
    ax.set_ylabel('F1 Accuracy Scores')
    ax.set_xlabel('Type of Classifier')
    ax.set_title('Accuracy Scores of Test Data for Ensemble Classifiers')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    fig.tight_layout()'''
    plt.show()

#bar graph to show different accuracy scores for each neural network
#using mean cross-val accuracy scores
def nnAccBar():
    labels = ['1', '2', '3', '4', '5', '6']
    ogAcc = [0.5991, 0.5997, 0.5319, 0.5866, 0.6043, 0.6053]
    osAcc = [0.5898, 0.5998, 0.5464, 0.5135, 0.3430, 0.5569]
    smAcc = [0.2205, 0.2311, 0.1995, 0.2271, 0.2337, 0.1312]
    
    x = np.arange(len(labels)) #label locations
    width = 0.2 #width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, ogAcc, width, label='Orginal dataset')
    rects2 = ax.bar(x + width, osAcc, width, label = 'Oversampled dataset')
    rects3 = ax.bar(x + width*2, smAcc, width, label = 'SMOTE Oversampling')
    
    #adding text and such
    ax.set_ylabel('Mean Accuracy Scores')
    ax.set_xlabel('Model Number')
    ax.set_title('Mean Accuracy Scores for Neural Network Classifiers')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower left") #bbox_to_anchor=(0.5, 1.15)
    
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    #fig.tight_layout()
    plt.show()
    
#essay and theme visualizations
#similar to a calender plot but for literature (paragraph and phrase instead of month and day)
#adapted from https://dzone.com/articles/plotting-a-calendar-in-matplotlib
def waff(df_waff, yc, xc):
    plt.figure(figsize=(15,10))
    ax = plt.gca().axes
    colors = {'none': 'gray', 'strong federal government': 'red', 'legislative': 'orange', 
              'nationalist outlook': 'yellow', 'weakness of articles of Confederation': 'green',
              'judicial': 'cyan', 'federalism': 'blueviolet', 'natural rights': 'fuchsia',
              'union': 'pink', 'human nature': 'maroon', 'interest-group theory': 'teal',
              'republican principles': 'lime', 'checks and balances': 'powderblue', 'executive': 'blue',
              'separation of powers': 'deeppink'}
    #adds the boxes
    for x in range(len(df_waff)):
        ax.add_patch(Rectangle(df_waff['Coordinates'][x], width=.8, height=.8,
                           color=colors[df_waff['Theme'][x]], alpha=1.0))
    #remember to add 1 
    plt.yticks(np.arange(1, yc))
    plt.xticks(np.arange(1, xc))
    plt.xlim(1, xc)
    plt.ylim(1, yc)
    plt.gca().invert_yaxis()
    # remove borders and ticks
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    #plt.tick_params(top=False, bottom=False, left=False, right=False)
    plt.title('Determined by LinearSVC with Original Data', fontsize=12)
    plt.suptitle('Distribution of Themes in Federalist No. 55', y=0.95, fontsize=18) 
    ax.set_ylabel('Paragraph Number')
    ax.set_xlabel('Phrase Number')
    # from https://stackoverflow.com/questions/31303912/matplotlib-pyplot-scatterplot-legend-from-color-dictionary 
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
    plt.legend(markers, colors.keys(), numpoints=1, loc='upper right', bbox_to_anchor=(1, 1.0))
    plt.show()

#create dataset that will work bets for DIY waffle/calnder chart chart
def waffData():
    filex = open('/mnt/linuxlab/home/r21sscott/REU_Project/FEDERALIST No. 55.txt', 'r') #filepath depends on user
    rawtxt = filex.read()
    filex.close()
    #get paragraph number
    pars = rawtxt.split('\n\n')
    for x in range(len(pars)):
        pars[x] = pars[x].replace('\n', ' ')
        pars[x] = pars[x].replace('"', '')
        pars[x] = pars[x].replace('.','')
    pars.remove('')
    pars.remove(' ')
    yc = len(pars) +1
    #print(len(pars))
    #print(pars)#numner of pars
    #get phrase number 
    df1 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeLinearSVC') #for labeled data use visData1 csv file
    currentPaper = []
    themes = []
    paperNum = []
    #gets docouments and themes for specific paper
    for x in range(len(df1)):
        if df1['Paper'][x] == 55:
            c = df1['Document'][x]
            c =c.replace('"', '')
            c = c.strip()
            currentPaper.append(c)
            themes.append(df1['Theme'][x])
            paperNum.append(df1['Paper'][x])
    #print(len(currentPaper))
    #make new dataframe for waffle data
    df2 = pd.DataFrame(data=currentPaper, columns=['Document']) 
    df2['Theme'] = themes
    df2['Paper'] = paperNum
    #print(len(df2)) #number of docs
    #get coordinates of paragrph and doc #
    coord = []
    for x in range(len(df2)):
        for y in range(len(pars)):
            if df2['Document'][x] in pars[y]: #works if they are exactly the same
                coord.append(((x+1),y+1))
                break
            elif pars[y] in df2['Document'][x]: 
                coord.append(((x+1),y+1))
                break

    #convert coordinates to make more whole/together
    i =1
    xc=1
    for x in range(len(coord)):
        if x == len(coord)-1:
            temp = (1, coord[x][1])
            coord[x] = temp
        elif coord[x][1] == coord[x+1][1]:
            temp = (i, coord[x][1])
            coord[x] = temp
            if int(temp[0]) > xc:
                xc = temp[0]
            i= i +1
        elif coord[x][1] != coord[x+1][1] and coord[x][1] == coord[x-1][1]:
            temp = (i, coord[x][1])
            coord[x] = temp
            if int(temp[0]) > xc:
                xc = temp[0]
            i=1
        else:
            temp = (1, coord[x][1])
            coord[x] = temp
    #print(len(coord))
    xc = xc+1
    df2['Coordinates'] = coord
    #print(df2['Theme'].value_counts())
    return df2, yc, xc
            
    #print(df2.head())
#a chart to compare the predictions from two different models
def compWaff(df_waff, yc, xc):
    df1 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeLinearSVC')
    df2 = pd.read_csv('/mnt/linuxlab/home/r21sscott/REU_Project/analyzeRandFor')
    plt.figure(figsize=(15,10))
    ax = plt.gca().axes
    colors = {'same':'purple', 'different': 'yellow'}
    for x in range(len(df1)):
        if df1['Theme'][x] == df2['Theme'][x]:
            ax.add_patch(Rectangle(df_waff['Coordinates'][x], width=.8, height=.8,
                                   color='purple', alpha=1.0))
        else:
            ax.add_patch(Rectangle(df_waff['Coordinates'][x], width=.8, height=.8,
                                   color='yellow', alpha=1.0))
    plt.yticks(np.arange(1, yc))
    plt.xticks(np.arange(1, xc))
    plt.xlim(1, xc)
    plt.ylim(1, yc)
    plt.gca().invert_yaxis()
    # remove borders and ticks
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    #plt.tick_params(top=False, bottom=False, left=False, right=False)
    plt.title('Comparing RandomForestClassifier and LinearSVC models', fontsize=14)
    plt.suptitle('Comparison of Model Results', y=0.95, fontsize=18) 
    ax.set_ylabel('Paragraph Number')
    ax.set_xlabel('Phrase Number')
    # from https://stackoverflow.com/questions/31303912/matplotlib-pyplot-scatterplot-legend-from-color-dictionary 
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
    plt.legend(markers, colors.keys(), numpoints=1, prop={'size': 12}, loc='best') #bbox_to_anchor=(0.75, 1.0
    plt.show()

main()