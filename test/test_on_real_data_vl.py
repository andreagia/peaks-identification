
import numpy as np
import pandas as pd

from core.peak_manager_V2 import Spectra, PeakManager, distance_matrix, assignmentToDf
from utils.peacks_testannotatedV2 import TestAnnotated
from peaksIdentification.postprocessing import sliding_avg


def getdistanze(spettro1, spettro2):
    spettr1_keys = spettro1.index.tolist()
    spettr2_keys = spettro2.index.tolist()
    cost = 0.

    dist_dict = {}
    for key1 in spettr1_keys:
        xy1 = spettro1.loc[key1].to_numpy()
        #xy1[1] *= 0.2

        if key1 in spettr2_keys:
            xy2 = spettro2.loc[key1].to_numpy()
            #xy2[1] *= 0.2
            print(key1, xy1, xy2)
            print(((xy1 - xy2)**2))
            dist = np.sqrt( ((xy1 - xy2)**2).sum() )
            print("dist ", dist)
            dist_dict[key1] = dist
            cost += dist

    return cost, dist_dict

def dict_to_df(dictionary):
    keys = []
    data = []
    for key, coordinate in dictionary.items():
        keys.append(key)
        data.append(coordinate)
    return pd.DataFrame(index=keys, columns=['x', 'y'], data=np.array(data, dtype=float))

def interpolate(aa_idx, df):
    avgs = []
    next_idx, prev_idx = None, None
    order = df['order'].tolist()
    #print(aa_idx)
    #print("order ", order)

    if order.index(aa_idx) < len(df) - 2:
        next_idx = order[order.index(aa_idx) + 1]
        next_avg = df[df['order'] == next_idx]['avg_window'][0]
        avgs.append(next_avg)

    if order.index(aa_idx) > 0:
        prev_idx = order[order.index(aa_idx) - 1]
        prev_avg = df[df['order'] == prev_idx]['avg_window'][0]
        avgs.append(prev_avg)

    #print("indices are ", next_idx, prev_idx)
    #print(avgs)
    approx_avg = float(sum(avgs))/len(avgs)

    return approx_avg

#list_files =["MMP12Cat_AHA_T1_ref_270510", "MMP12Cat_Dive_T1_ref_peaks"]
#list_files =["MMP12Cat_NNGH_T1_ref_300707", "MMP12Cat_Dive_T1_ref_peaks"]
list_files =["MMP12Cat_AHA_T1_ref_270510","MMP12Cat_NNGH_T1_ref_300707"]

# list_files =["MMP12_AHA_ref","MMP12Cat_NNGH_T1_ref_300707"] # 1 test
#list_files=["MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref"] # 2 test
#list_files = ["MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510"]

ta = TestAnnotated()
ass1, ass2 = ta.getAssignmentData(list_files[0],list_files[1])

peaks = dict_to_df(ass1) # to dataframe
new_peaks = dict_to_df(ass2) # to dataframe
print(peaks)
print(new_peaks)

old_spectra = Spectra( peaks )
new_spectra = Spectra( new_peaks )

pm = PeakManager(search_depth=1, max_search_per_level=2, log=False)
# peaks_assignment = pm.assign(peaks, new_peaks)
# peaks_assignment = pm.assign(old_spectra, new_spectra)
# print("BEST ASSIGNMENT COST IS: ", peaks_assignment.cost)

xy_free, xy_with_ligands, associations, _ = pm.getAssociations(old_spectra, new_spectra)

#***********************************************************************************************************************
#***********************************************************************************************************************

target_distances = {}
distances = distance_matrix(old_spectra.xy(), new_spectra.xy())
dataframe = pd.DataFrame(data = distances, index=old_spectra.keys(), columns=new_spectra.keys())

for k0 in old_spectra.keys():
    if k0 in new_spectra.keys():
        target_distances[k0] = dataframe.loc[k0][k0]

# print(target_distances)

df1 = assignmentToDf(associations, target_distances)


distances = df1['Distance'].tolist()
sl_avg = sliding_avg(distances, half_window_size=3)
df1['window_avg'] = sl_avg

print("df1 = assignmentToDf(associations, target_distances)")
print(df1)

#plotplot(old_spectra, new_spectra, associations)
#plotHistogram(df1, real_dist_dict = target_distances)
### sl_avg, aa_names, _order = plotHistogram(df1, real_dist_dict = target_distances)
### plotHistogram(associations, real_dist_dict = real_dist_dict)

print("="*200)
print("="*200)

########################################################################################################################
########################################################################################################################
# POST PROCESSING STEP:
# per i picchi non coinvolti nell assegnamento non e stata calcolata la media dell intorno
# li ri-consideriamo  assegnandogli la media delle medie dei vicini (intanto a tutti gli aminoacidi assegniamo w_avg = -1)
peaks['avg_window'] = [-1]*len(peaks)

 #- a quelli per cui è stata calcolara, gli assegniamo il valore vero
aa_names = df1['Name'].tolist()
for u, aa in zip(sl_avg, aa_names):
    peaks.loc[aa, 'avg_window'] = u

#- ai rimanenti (quelli con w_avg < 0) gli diamo quella dei vicini, interpolata
#print(peaks[peaks['avg_window']<0])

# qui possiamo fare meglio..!!
f = lambda x : int(''.join([c for c in x if c.isnumeric()]))
ll = peaks.index.tolist()
order = list(map(f, ll))
peaks['order']=order
peaks = peaks.sort_values(by=['order'])
peaks['assignedTo'] = ['NotAssigned']*len(peaks)

for item in associations:
    source_, dest_, dist_ = item
    peaks.loc[source_, 'assignedTo'] = dest_

___i = peaks[peaks['avg_window']<0].index.tolist()
for idx in peaks[peaks['avg_window']<0].index.tolist():
    aa_idx = peaks.loc[idx]['order']
    approx_avg = interpolate(aa_idx, peaks)
    # print("///", approx_avg)
    peaks.loc[idx, 'avg_window'] = approx_avg
# print(peaks.loc[___i, :])

########################################################################################################################
# Fixing dei picchi assegnati precedentemente...
# LA MATRICE DELLE DISTANZE DEVE ESSERE CALCOLATA UNA VOLTA SOLA, ALL'INIZIO. FINE!
# Una volta calcolata, si modificano le distanze calcolate (as Antonio's way)

# questo lo sto copiando da peak_manager_V2.py riga 256

assignedTo = peaks['assignedTo'].tolist()
print(assignedTo)
print(new_peaks.index.tolist())
col_idxs = [new_peaks.index.tolist().index(x)
            for x in assignedTo if x != 'NotAssigned']
#for i, j in enumerate(col_idxs):
    #distance_matrix[i, j] /= deltas[i]
#poi passare in input questa nuova distance_matrix

import sys
sys.exit()

col_idxs = [ list(new_peaks.keys()).index(x) for x in assignedTo if x != 'NotAssigned']

for i, j in enumerate(col_idxs):
    print(i, j)
    distance_matrix[i, j] /= deltas[i]




#peaks = peaks.sort_index()
print(peaks)
print(new_peaks)

old_spectra1 = Spectra( peaks )
new_spectra1 = Spectra( new_peaks )

_, _, associations = pm.getAssociations(old_spectra1, new_spectra1)

#plotplot(old_spectra, new_spectra, associations)
#plotHistogram(associations, real_dist_dict = target_distances)