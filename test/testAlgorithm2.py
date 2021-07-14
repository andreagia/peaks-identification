import utils.peacks_testannotatedV2 as pk
import pandas as pd
import numpy as np
import math
import random
from core.peak_manager_V2 import distance_matrix


def dict_to_df(dictionary):
    keys = []
    data = []
    for key, coordinate in dictionary.items():
        keys.append(key)
        data.append(coordinate)
    return pd.DataFrame(index=keys, columns=['x', 'y'], data=np.array(data, dtype=float))


def cust_window_avg(X_, mask, idx, size=2):

    left_idx = max(idx - size, 0)
    right_idx = min(idx + size + 1, len(X_))
    w = X_[left_idx:right_idx]

    #print(w)
    #print(mask[left_idx:right_idx])
    w = w*mask[left_idx:right_idx]
    denominator = sum(mask[left_idx:right_idx])
    #print("(((", w)
    #print("(((", sum(w))
    #print("(((", denominator)
    #print("=>", sum(w) / denominator)

    if denominator > 0:
        return sum(w) / denominator
    else:
        return 0.


def sliding_avg(x, half_window_size=2):
    assert isinstance(x, list)
    X_ = x

    def window_avg(idx, size=half_window_size):
        left_idx = max(idx - size, 0)
        right_idx = min(idx + size + 1, len(x))
        w = X_[left_idx:right_idx]
        # print(x)
        # print(w, " --> ", sum(w) / len(w))
        return sum(w) / len(w)

    pos = range(len(X_))
    rr = map(window_avg, pos)
    return list(rr)

#######################################################################################
# Preprocessing

ta = pk.TestAnnotated()
#f1 ,f2 = ta.getAssignmentData("CAIIZn_000_furo_03_170317","CAIIZn_100_furo_11_170317")
peaks, new_peaks = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510")
peaks, new_peaks = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref")

peaks = dict_to_df(peaks) # to dataframe
new_peaks = dict_to_df(new_peaks) # to dataframe


peaks['order'] = peaks.index.map(
    lambda x : int(''.join([c for c in x if c.isnumeric()])))
peaks = peaks.sort_values(by=['order'])

new_peaks['order'] = new_peaks.index.map(
    lambda x : int(''.join([c for c in x if c.isnumeric()])))
new_peaks = new_peaks.sort_values(by=['order'])

#######################################################################################

Distances = distance_matrix(peaks[['x', 'y']].to_numpy(), new_peaks[['x', 'y']].to_numpy())
print(Distances)

peaks['assigned_to'] = ['na']*len(peaks)
peaks['dist'] = [-1]*len(peaks)
print(peaks)
print(new_peaks)

#x = assigned_aa_distances
def get_avg_distances(peaks):
    order = peaks['order'].tolist()
    d = peaks['dist'].tolist()
    print(order)
    print(d)

    m = np.array(d) >= 0
    res = []

    for i in order:
        idx = order.index(i)
        i_avg = cust_window_avg(d, m, idx, size=10)
        res.append(i_avg)
    return np.array(res)

def get_not_associated_keys(peaks):
    peaks[peaks['assigned_to'] == 'na'].index.tolist()

cost_v = []

for h in range(1):
    print("##"*50)

    # devo iterare sui singoli picchi...

    peaks_ = peaks.index.tolist()
    idxs_ = list(range(len(peaks_)))
    random.shuffle(idxs_)
    peaks_ = list(np.array(peaks_)[idxs_])

    peaks_keys = peaks.index.tolist()
    new_peaks_keys = new_peaks.index.tolist()


    for iter in range(min(len(peaks), len(new_peaks))):
        ######
        na_peaks_keys = peaks[peaks['assigned_to'] == 'na'].index.tolist()
        na_peaks_keys_idx = [peaks_keys.index(x) for x in na_peaks_keys]
        ass_new_peaks_keys = peaks[peaks['assigned_to'] != 'na']['assigned_to'].tolist()
        na_new_peaks_keys = [x for x in new_peaks_keys if x not in ass_new_peaks_keys]
        na_new_peaks_keys_idx = [new_peaks_keys.index(x) for x in na_new_peaks_keys]
        ######

        x_avg = get_avg_distances(peaks)
        Dist2 = np.abs(Distances - x_avg.reshape(-1, 1))

        tmp_best_idxs = np.argmin(Dist2, axis=1) # estraggo gli indici dei migliori per ogni picco
        tmp_best_vals = np.min(Dist2, axis=1) # estraggo le relative distanze

        # seleziono la coppia di picchi a distanza minima
        best_Pi_idx, best_Sj_idx, best_dist_ij = -1, -1, math.inf
        for Pi_idx in na_peaks_keys_idx:
            for Sj_idx in na_new_peaks_keys_idx:
                dist_ij = Dist2[Pi_idx, Sj_idx]
                if dist_ij < best_dist_ij:
                    best_Pi_idx = Pi_idx
                    best_Sj_idx = Sj_idx

        P_key = peaks_keys[best_Pi_idx]
        peaks.loc[P_key, 'dist'] = i_j_dist2
        peaks.loc[P_key, 'assigned_to'] = Pj_new_key2


    #for i, P_key in enumerate(peaks.index.tolist()):
    for i, P_key in zip(idxs_, peaks_):

        na_peaks_keys = peaks[peaks['assigned_to'] == 'na'].index.tolist()
        na_peaks_keys_idx = [peaks_keys.index(x) for x in na_peaks_keys]

        ass_new_peaks_keys = peaks[peaks['assigned_to'] != 'na']['assigned_to'].tolist()
        na_new_peaks_keys = [x for x in new_peaks_keys if x not in ass_new_peaks_keys]
        na_new_peaks_keys_idx = [new_peaks_keys.index(x) for x in na_new_peaks_keys]

        print("P key: ", P_key)
        x_avg = get_avg_distances(peaks)
        #print("x_avg: ", x_avg)

        Dist2 = np.abs(Distances - x_avg.reshape(-1,1))

        print("na_peaks_keys_idx: ", na_peaks_keys_idx)
        print("na_new_peaks_keys_idx: ", na_new_peaks_keys_idx)


        #se prendiamo solo le distanze dei non assegnati??
        #subDist2 = Dist2[na_peaks_keys_idx, :][:,na_new_peaks_keys_idx]

        #tmp_best_idxs2 = np.argmin(subDist2, axis=1) # estraggo gli indici dei migliori per ogni picco
        #tmp_best_vals2 = np.min(subDist2, axis=1) # estraggo le relative distanze
        #sub_idx = na_peaks_keys_idx.index(i)
        #i_j_dist2 = tmp_best_vals2[sub_idx]  # distance dell'assegnamento           # ok
        #Pj_new_sub_idx = tmp_best_idxs2[sub_idx]  # (new) PEAK j assigned to i      # ok
        #Pj_new_idx2 = na_new_peaks_keys_idx[Pj_new_sub_idx]
        #Pj_new_key2 = new_peaks.index.tolist()[Pj_new_idx2]                         # ok


        tmp_best_idxs = np.argmin(Dist2, axis=1) # estraggo gli indici dei migliori per ogni picco
        tmp_best_vals = np.min(Dist2, axis=1) # estraggo le relative distanze

        #Pi_old_idx = np.argmin(tmp_best_vals) # PEAK i (old) a cui è stato associato il corrispettivo
        #i_j_dist = tmp_best_vals[Pi_old_idx]  # distance dell'assegnamento
        #Pj_new_idx = tmp_best_idxs[ Pi_old_idx] # (new) PEAK j assigned to i
        i_j_dist = tmp_best_vals[i]  # distance dell'assegnamento           #<--
        Pj_new_idx = tmp_best_idxs[i]  # (new) PEAK j assigned to i         #<--
        Pj_new_key = new_peaks.index.tolist()[Pj_new_idx]                   #<--


        possible_assoc_for_i = []
        for j, dist in enumerate(Dist2[i]):
            possible_assoc_for_i.append((dist, j))
            
        #sort(possible_assoc_for_i by 0)
        possible_assoc_for_i.sort(key=lambda x: x[0])

        for dist, j in possible_assoc_for_i:
            if j in na_new_peaks_keys_idx:
                i_j_dist2 = dist
                Pj_new_idx2 = j
                break
                
        Pj_new_key2 = new_peaks.index.tolist()[Pj_new_idx2]


        #se Pj_new_key è stato già assegnato ???
        #si valuta lo scambio con chi ce l ha. In caso di scambio, al quell altro che gli diamo ??? deve cercarne un altro... e se pure quello è occupato ???
        #allora, la cosa e


        print("Pj_new_idx {}, Pj_new_key {}".format(Pj_new_idx, Pj_new_key))
        #print(tmp_best_idxs)
        #print(tmp_best_vals)
        #print(peaks.loc[P_key])
        print("i_j_dist", i_j_dist)

        peaks.loc[P_key, 'dist'] = i_j_dist2
        peaks.loc[P_key, 'assigned_to'] = Pj_new_key2
        #print(np.min(Dist2, axis=1))
        #print(tmp_best.shape)
        #print(new_peaks.iloc[tmp_best].index.tolist())

        #peaks['dist'] = Distances[range(len(peaks)), tmp_best_idxs]
        #peaks['dist'] = Dist2[range(len(peaks)), tmp_best]

        #peaks['assigned_to'] = new_peaks.iloc[tmp_best_idxs].index.tolist()

        #print(peaks)
    cost = np.abs(np.array(peaks['dist'].tolist()) - x_avg).sum()
    print(peaks)

    s1 = peaks.index.tolist()
    s2 = peaks['assigned_to'].tolist()
    acc = (np.array(s1) == np.array(s2)).sum()/len(s1)
    print(acc)
    print(len(s2), len(set(s2)))
    print("\n====>", cost)
    cost_v.append(acc)
    print("//"*50)


#plt.plot(range(len(cost_v)), cost_v)
#plt.show()