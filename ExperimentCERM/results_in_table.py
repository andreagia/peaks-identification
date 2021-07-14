import numpy as np
import pandas as pd
import math
import pathlib

from utils.peacks_testannotatedV2 import TestAnnotated as DataParser
from core.peak_manager_V2 import Spectra, PeakManager
from core.peak_manager_V2 import distance_matrix as mdistance_matrix
from core.neighbor_based_assignment import estimate_assignment as neighbor_algorithm, get_avg_distances
from core.operation_research_tool import optimize_assegnation
from data import data_info
import copy

parser = DataParser()


def dict_to_df(dictionary):
    keys = []
    data = []
    for key, coordinate in dictionary.items():
        keys.append(key)
        data.append(coordinate)
    return pd.DataFrame(index=keys, columns=['x', 'y'], data=np.array(data, dtype=float))


def order_dataframe(peaks):
    peaks['order'] = peaks.index.map(lambda x: int(''.join([c for c in x if c.isnumeric()])))
    peaks = peaks.sort_values(by=['order'])
    return peaks


def vl_algorithm1(peaks, new_peaks):
    old_spectra = Spectra(peaks)
    new_spectra = Spectra(new_peaks)
    pm = PeakManager(search_depth=1, max_search_per_level=2, log=False)
    #pm = PeakManager(search_depth=0, max_search_per_level=2, log=False)
    xy_free, xy_with_ligands, associations, accuracy = pm.getAssociations(old_spectra, new_spectra)
    return accuracy


def vl_second_algorithm(peaks, new_peaks):
    best_acc, best_w_size = 0., -1

    for half_size in range(1, 11):
        print("half_size ", half_size)
        print(peaks)
        peaks, new_peaks = neighbor_algorithm(peaks, new_peaks, w_half_size=half_size)
        x_avg = get_avg_distances(peaks, w_half_size=half_size)
        print(peaks)
        print(new_peaks)
        #print("---->", x_avg)
        #sys.exit()
        cost = np.abs(np.array(peaks['dist'].tolist()) - x_avg).sum()
        # print(peaks)

        assigned_peaks = peaks[peaks['assigned_to']!='na']
        s1 = assigned_peaks.index.tolist()
        s2 = assigned_peaks['assigned_to'].tolist()

        good, wrong = 0., 0.
        for j in range(len(s1)):
            if s1[j] == s2[j]:
                good+=1
            else:
                wrong+=1
        acc = good/(good+wrong)
        #print("?", good, wrong, "acc", acc, "TOT: ", good+wrong)
        #print("Cost = ", sum(assigned_peaks['dist'].tolist()),
        # sum(assigned_peaks['dist'].tolist())/len(assigned_peaks))
        #print(len(peaks), len(new_peaks), len(peaks[peaks['assigned_to']=='na']))
        #acc = (np.array(s1) == np.array(s2)).sum() / len(s1)
        print("half_size ",half_size, "acc ", acc)
        if acc > best_acc:
            best_acc = acc
            best_w_size = half_size
    #print("Accuracy: ", acc)
    #print(len(s2), len(set(s2)))
    #print("\====>", cost)
    print("half size: ", best_w_size)
    return best_acc


def optimizer_algorithm(peaks, new_peaks):

    if len(peaks) < len(new_peaks):
        peaks0 = peaks
        new_peaks0 = new_peaks
        peaks = new_peaks0
        new_peaks = peaks0

    peaks = order_dataframe(peaks)
    new_peaks = order_dataframe(new_peaks)

    DistMatrix = mdistance_matrix(peaks[['x', 'y']].to_numpy(), new_peaks[['x', 'y']].to_numpy())
    _cost, _assegnations, __acc = optimize_assegnation(peaks, new_peaks, DistMatrix)

    #print("Cost: ", _cost)
    good, wrong = 0., 0.
    for i, j, d in _assegnations:
        # print(item)
        if peaks.index.tolist()[i] == new_peaks.index.tolist()[j]:
            good += 1
        else:
            wrong += 1
    acc = good / (good + wrong)
    #print("AAAAA", __acc, acc)

    #print("Good: ", good, "Wrong: ", wrong, "Acc: ", float(good) / (good + wrong))
    return acc


def mediaMobile(vettore, w_half_size):
    res = []
    for idx in range(len(vettore)):
        left_idx = max(0, idx-w_half_size)
        right_idx = min(len(vettore), idx+w_half_size+1)

        w = []
        for j in range(left_idx, right_idx):
            w.append(vettore[j])
        res.append(sum(w)/len(w))

    return np.array(res)


def mediaMobile2(vdistance):
    meanv = []
    for i in range(len(vdistance)):
        st = []
        # print("new")
        for a in range(5):
            a -= 2
            # print(a, i+a)
            if a + i >= 0 and a + i < len(vdistance):
                # if a+i >= 0 and a+i < len(vdistance) and a != 0:
                st.append(vdistance[a + i])
        nst = np.array(st)
        # print(nst,nst.mean())
        meanv.append(nst.mean())
    return np.array(meanv)


def optimizer_with_postprocessing(peaks, new_peaks):

    if len(peaks) < len(new_peaks):
        peaks0 = peaks
        new_peaks0 = new_peaks
        peaks = new_peaks0
        new_peaks = peaks0

    peaks = order_dataframe(peaks)
    new_peaks = order_dataframe(new_peaks)

    best_cost = math.inf
    best_acc = 0.
    best_w_half_size = None
    best_attempt = None

    #for w_half_size in range(2):
    w_half_size = 3
    print("/// w_half_size", w_half_size)

    DistMatrix = mdistance_matrix(peaks[['x', 'y']].to_numpy(), new_peaks[['x', 'y']].to_numpy())
    _cost, _assegnations, _acc = optimize_assegnation(peaks, new_peaks, distances=DistMatrix)
    #print("_assegnations", _assegnations)
    #print(_acc)
    if _acc > best_acc:
        best_acc = _acc
        best_w_half_size = w_half_size
        best_cost = _cost
        best_attempt = 0

    for w_half_size in range(1, 10):
        DistMatrix2 = copy.deepcopy(DistMatrix)
        for attempt in range(5):
            #print(peaks)

            avg = get_avg_distances(peaks[peaks['Distance']>-1], w_half_size=w_half_size, distance_col_name="Distance")
            #avg1 = mediaMobile(peaks['Distance'].tolist(), w_half_size=2)
            #avg2 = mediaMobile2(peaks['Distance'].tolist())

            #print("len(avg)", len(avg))
            #print("len(_assegnations)", len(_assegnations))

            #print(peaks['Distance'].tolist())
            #print(list(avg))
            #print(list(avg1))
            #print(list(avg2))
            for jj, (ai, _, _ ) in enumerate(_assegnations):
                DistMatrix2[ai, :] = DistMatrix2[ai, :] / (avg[jj])

            _cost, _assegnations, _acc = optimize_assegnation(peaks, new_peaks, distances=DistMatrix2)
            #print("attempt", attempt, _acc)

            if _acc > best_acc:
                best_acc = _acc
                best_w_half_size = w_half_size
                best_cost = _cost
                best_attempt = attempt

    #print("Good: ", good, "Wrong: ", wrong, "Acc: ", float(good) / (good + wrong))
    return best_acc


experiments =[
    ["MMP12Cat_AHA_T1_ref_270510", "MMP12Cat_Dive_T1_ref_peaks"], # OK

    ["MMP12Cat_AHA_T1_ref_270510","MMP12Cat_NNGH_T1_ref_300707"], # OK

    ["MMP12_AHA_ref","MMP12Cat_NNGH_T1_ref_300707"], # OK

    ["MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref"], # OK

    ["MMP12_AHA_ref", "MMP12Cat_Dive_T1_ref_peaks"],

    ["MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510"],

    ["CAIIZn_000_furo_03_170317", "CAIIZn_100_furo_11_170317"],

    ["CAIIZn_100_furo_11_170317", "CAIIZn_000_furo_03_170317"],

    ["CAIIZn_0.00_sulpiride_03_040417", "CAIIZn_5mM_sulpiride_19_040417"],

    ["CAII_Zn_000_pTulpho_03_220317", "CAII_Zn_100f_pTulpho_18_220317"],

    ["CAII_Zn_000_pTS_04_291216", "CAII_Zn_100_pTS_15_291216"],

    ["CAII_Zn_000_oxalate_04_221116", "CAII_Zn_15mM_oxalate_31_221116"],
]

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

experiments_dundee = [
    ['Ube2T_ref_final.csv', 'ube2t_em02_5mM_final.csv']
]

algorithms = [
    ("vl_first", vl_algorithm1),
    ("vl_second", vl_second_algorithm),
    ("optimizer", optimizer_algorithm),
    ("optimizer+post_proc", optimizer_with_postprocessing)
]

#result_df = pd.DataFrame(columns=['Free', 'With_ligand', 'vl_first', 'vl_secong', 'optimizer', 'optimizer+post_proc'])
#result_df = pd.DataFrame(columns=['Free', 'With_ligand', 'optimizer', 'optimizer+post_proc'])

def CERM_data_run():
    for prot_ligand in experiments:
        assert len(prot_ligand) == 2
        print("==> FREE: ", prot_ligand[0], "WITH LIGAND: ", prot_ligand[1])

        algs_results = {}
        for alg_name, algorithm in algorithms:
            peaks, new_peaks = parser.getAssignmentData(prot_ligand[0], prot_ligand[1])
            peaks = dict_to_df(peaks)  # to dataframe
            new_peaks = dict_to_df(new_peaks)  # to dataframe
            dd = {'Free': prot_ligand[0], 'With_ligand': prot_ligand[1]}

            accuracy = algorithm(peaks, new_peaks)
            print("[[{}]]".format(alg_name), accuracy)
            algs_results[alg_name] = accuracy
        result_df = result_df.append({**dd, **algs_results}, ignore_index=True)
        print()
    result_df.to_csv("peak_shift_resultsV2.csv")

def csv_to_df(csv_file):
    data = pd.read_csv(csv_file)
    data.rename(columns={data.columns[0]: "Number", data.columns[1]: "x", data.columns[2]: "y"}, inplace=True)
    data.set_index("Number", inplace=True)
    data.index = data.rename(index=str).index
    #print(data)
    #print(data.index)
    return data

def Dundee_data_run():
    result_df = pd.DataFrame(
        columns=['Free', 'With_ligand', 'vl_first', 'vl_secong', 'optimizer', 'optimizer+post_proc'])

    this_path = pathlib.Path(__file__).resolve().parent

    for prot_ligand in experiments_dundee:
        print(prot_ligand)
        assert len(prot_ligand) == 2
        print("==> FREE: ", prot_ligand[0], "| WITH LIGAND: ", prot_ligand[1])
        #data_path = '../data/dundee/'
        data_path = data_info.dundee_data_path
        algs_results = {}
        for alg_name, algorithm in algorithms:

            #peaks, new_peaks = parser.getAssignmentData(prot_ligand[0], prot_ligand[1])
            #peaks = dict_to_df(peaks)  # to dataframe
            #new_peaks = dict_to_df(new_peaks)  # to dataframe

            print(data_path)
            #print(data_path + prot_ligand[0])
            print(data_path.joinpath(prot_ligand[0]))

            peaks = csv_to_df(data_path.joinpath(prot_ligand[0]))
            #new_peaks = csv_to_df(data_path + prot_ligand[1])
            new_peaks = csv_to_df(data_path.joinpath(prot_ligand[1]))

            dd = {'Free': prot_ligand[0], 'With_ligand': prot_ligand[1]}

            accuracy = algorithm(peaks, new_peaks)
            print("[[{}]]".format(alg_name), accuracy)
            algs_results[alg_name] = accuracy
        result_df = result_df.append({**dd, **algs_results}, ignore_index=True)
        print()

    result_df.to_csv(
        this_path.joinpath( "dundee_peak_shift_resultsV2.csv"))

#CERM_data_run()
Dundee_data_run()