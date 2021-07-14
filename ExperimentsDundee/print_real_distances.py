
from data import data_info
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def csv_to_df(csv_file):
    data = pd.read_csv(csv_file)
    data.rename(columns={data.columns[0]: "Number", data.columns[1]: "x", data.columns[2]: "y"}, inplace=True)
    data.set_index("Number", inplace=True)
    data.index = data.rename(index=str).index
    #print(data)
    #print(data.index)
    return data

data_path = data_info.dundee_data_path

experiments_dundee = [
    ['Ube2T_ref_final.csv', 'ube2t_em02_5mM_final.csv'],
    ['Ube2T_ref_final.csv', 'ube_2t_em11_3mM_final.csv'],
    ['Ube2T_ref_final.csv', 'ube2t_em04_3mM_final.csv'],
    ['Ube2T_ref_final.csv', 'ube2t_em09_3mM_final.csv'],
    ['Ube2T_ref_final.csv', 'ube2t_em17_3mM_final.csv'],
    ['Ube2T_ref_final.csv', 'ube2t_em29_3mM_final.csv'],

    ['baz2b_phd_ref_renumbered.csv', 'baz2b_vs_5-mer_20_1_renumbered.csv'],
    ['baz2a_phd_ref_renumbered.csv', 'baz2a_phd_vs_5mer_64_1_renumbered.csv'],
    ['baz2a_phd_ref_renumbered.csv', 'baz2a_phd_vs_10mer_8_1_renumbered.csv'],

]

for free_prot, prot_with_ligand in experiments_dundee:
    peaks = csv_to_df(data_path.joinpath(free_prot))
    new_peaks = csv_to_df(data_path.joinpath(prot_with_ligand))
    print(peaks)
    print(new_peaks)

    peak_distances = pd.DataFrame(columns=['shift'])

    for free_pk_idx in peaks.index:
        if free_pk_idx in new_peaks.index:
            free_xy = peaks.loc[free_pk_idx].to_numpy()
            with_ligand_xy = new_peaks.loc[free_pk_idx].to_numpy()

            free_prot_name = free_prot.split('.csv')[0]
            with_ligand_name = prot_with_ligand.split('.csv')[0]

            print(free_pk_idx, free_xy, with_ligand_xy,
                  free_xy - with_ligand_xy,
                  np.square(free_xy - with_ligand_xy),
                  np.square(free_xy - with_ligand_xy).sum(),
                  np.sqrt( np.square(free_xy - with_ligand_xy).sum()))

            dist = np.sqrt( np.square(free_xy - with_ligand_xy).sum() )
            peak_distances.loc[free_pk_idx, 'shift'] = dist
            #print(dist)
            print(np.linalg.norm(free_xy - with_ligand_xy))

    fig2 = go.Figure(layout=go.Layout(
        title= free_prot_name + " - " + with_ligand_name,
        font=dict(size=10)))
    fig2.add_trace(go.Bar(
        # x=df1.index,
        x=peak_distances.index.tolist(),
        y=peak_distances['shift'].tolist(),
        name='Real Shift Distance',
        marker_color='lightgray'
    ))
    fig2.show()
    #peak_distances.plot.bar()
    #plt.title(free_prot+" * "+prot_with_ligand)
    #ax1 = plt.axes()
    #x_axis = ax1.axes.get_xaxis()
    #x_axis.set_visible(False)
    #plt.show()


    print(peak_distances)
    peak_distances.to_csv(free_prot_name+"____"+with_ligand_name+".csv")

    sys.exit()