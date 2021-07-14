
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
# from scipy.spatial import distance_matrix
import plotly.graph_objects as go
from peaksIdentification.postprocessing import sliding_avg
import numpy as np


def plotplot(spettro_old, spettro_new, associations):

    print(associations)
    print(len(associations))
    """
    :param spettro_old: dataframe
    :param spettro_new: dataframe
    :param associations: [('k_old_1', 'k_new_1', d), ('k_old_2', 'k_new_2', d), ..., ('k_old_N', 'k_new_N', d),]
    """

    X = spettro_old.dd.to_numpy()
    keys_X = spettro_old.dd.index.tolist()

    Y = spettro_new.dd.to_numpy()
    keys_Y = spettro_new.dd.index.tolist()

    fig = go.Figure()
    fig.update_layout(width=1300, height=1000)
    fig['layout']['xaxis']['autorange'] = "reversed"
    fig['layout']['yaxis']['autorange'] = "reversed"

    fig.add_trace(
        go.Scatter(
            mode='markers+text',
            x=X[:, 0],
            y=X[:, 1]*0.2,
            marker=dict(
                color='LightSkyBlue',
                size=4,
                line=dict(
                    color='MediumPurple',
                    width=1
                )
            ),
            name='Spectra1',
            text=keys_X, textposition="bottom center"
        ))


    fig.add_trace(
        go.Scatter(
            mode='markers+text',
            x=Y[:, 0],
            y=Y[:, 1]*0.2,
            marker=dict(
                color='Coral',
                size=4,
                line=dict(
                    color='MediumPurple',
                    width=1
                )
            ),
            name='Spectra2',
            text=keys_Y, textposition="bottom center"
        ))

    fig.show()

    hist = []
    histn = []
    histi = []

    for triple in associations:
        # print(triple)
        key_old = triple[0]
        key_new = triple[1]
        dist = triple[2]
        hist.append(dist)
        histn.append(key_old)
        strnkey1 = ''.join(char for char in key_old if char.isnumeric())
        histi.append(int(strnkey1))

        old_p_xy = spettro_old.dd.loc[key_old].to_numpy(dtype=float)
        new_p_xy = spettro_new.dd.loc[key_new].to_numpy(dtype=float)

        ddist = np.sqrt(((old_p_xy - new_p_xy)**2).sum())
        #print("??", old_p_xy, new_p_xy, ddist)
        if key_old == key_new:
            color = "MediumPurple"
        else:
            color = "red"

        fig.add_trace(go.Scatter(x=[old_p_xy[0], new_p_xy[0]], y=[old_p_xy[1]*0.2, new_p_xy[1]*0.2],
                                 mode='lines',
                                 showlegend=False,
                                 text='provaaa',
                                 line=dict(color=color)))
    fig.show()

    '''
    df1 = pd.DataFrame({"DistanceC": hist, "Index":histi, "Name":histn})
    df1 = df1.sort_values(by=['Index'])
    print(df1)

    fig2 = px.bar(df1, x = "Name", y="DistanceC")
    fig2.show()
    '''


def plotHistogram(df1, real_dist_dict=None):
    #print("REAL DIST DICT ", real_dist_dict) # real distance dictionary
    #print("SIZE REAL DIST DICT ", len(real_dist_dict))
    #print("= = = = =>\n",df1)

    #distances = df1['Distance'].tolist()
    #sl_avg = sliding_avg(distances, half_window_size=3) # sliding window avg on our estimated shift distances
    #print(sl_avg)

    #print(len(df1['Name'].tolist()), df1['Name'].tolist())
    #print(len(df1['Index'].tolist()), df1['Index'].tolist())
    #print("==============")

    #df1['window_avg'] = sl_avg
    sl_avg = df1['window_avg']

    # plot dello istogramma
    #fig2 = px.bar(df1, x = "Name", y="Distance")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df1["Name"],
        y=df1["Distance"],
        name='DistanceGG',
        marker_color='blue'
    ))

    if real_dist_dict is not None:
        #fig2 = px.bar(df1, x="Name", y="Real_dist")
        fig2.add_trace(go.Bar(
            x=df1["Name"],
            y=df1["Real_dist"],
            name='Distance real',
            marker_color='lightgray'
        ))

    # plotta il punto indicante il valore della media mobile
    fig2.add_trace(
        go.Scatter(
            mode='markers+text+lines',
            x=df1['Name'],
            y=sl_avg,
            marker=dict(
                color='Coral',
                size=4,
                line=dict(color='MediumPurple',width=1
                )
            ),
            name='Window avg', text='', textposition="bottom center"
        ))

    fig2.show()
    #return sl_avg, df1['Name'].tolist(), df1['Index'].tolist()






