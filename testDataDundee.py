

import argparse
from datetime import datetime
import core.proxy as proxy
import core.graphics as graphics
from core.metrics import custom_accuracy
import sys
import pathlib
from utils.peacks_testannotatedV2 import TestAnnotated as DataParser
import configparser




def run_assignment(free_protein, with_ligand_protein, out_file, plot, algorithm):
    print("Free: ", free_protein)
    print("With ligand: ", with_ligand_protein)

    free_data_filename = pathlib.PurePath(free_protein).name.split('.csv')[0]
    withligand_data_filename = pathlib.PurePath(with_ligand_protein).name.split('.csv')[0]

    print("Base names: ", free_data_filename, withligand_data_filename)
    if out_file != None:
        output_filename = out_file
    else:
        output_filename = free_data_filename+"--"+withligand_data_filename+"--"+algorithm


    # RUN ALGORITHM
    assignemts, f_peaks, wl_peaks, ACC = proxy.estimate_shifts(
        free_protein,
        with_ligand_protein,
        assignment_algorithm=algorithm,
        cerm_data=False)

    # PROCESS RESULTS
    assignemts.index.name="ResidueKey"
    assignemts.to_csv(output_filename+"({:.2f}).csv".format(ACC*100), sep=';')


    P_key_list = assignemts.index.tolist()
    S_key_list = assignemts['assigned_to'].tolist()
    assert len(P_key_list) == len(S_key_list)
    wrong = 0.
    ok = 0.
    for j, _ in enumerate(P_key_list):
        if S_key_list[j] != 'na':
            if P_key_list[j] == S_key_list[j]:
                ok+=1
            else:
                wrong+=1
    acc = ok /(ok + wrong)

    print("ACC: ", ACC, "///", custom_accuracy(assignemts, 'assigned_to'), "[", acc, "]")
    #print("=>", output_filename, "[", acc, "]", )

    if plot == True:
        graphics.plotProfile(assignemts, free_protein, with_ligand_protein, acc=acc)
        graphics.plotPeaksShifts(f_peaks, wl_peaks, assignemts, free_protein, with_ligand_protein, acc=acc)

    print("fine")


def main():

    data_dundee = [
        ['Ube2T_ref_final.csv', 'ube2t_em02_5mM_manual.csv'],
        ['Ube2T_ref_final.csv', 'ube2T_em11_3mM_manual.csv'],
        ['Ube2T_ref_final.csv', 'ube2t_em04_3mM_manual.csv'],
        ['Ube2T_ref_final.csv', 'ube2t_em09_3mM_manual.csv'],
        ['Ube2T_ref_final.csv', 'ube2t_em17_3mM_final.csv'],
        ['Ube2T_ref_final.csv', 'ube2t_em29_3mM_manual.csv'],

        ['baz2b_phd_ref_renumbered.csv', 'baz2b_vs_5-mer_20_1_renumbered.csv'],
        ['baz2a_phd_ref_renumbered.csv', 'baz2a_phd_vs_5mer_64_1_renumbered.csv'],
        ['baz2a_phd_ref_renumbered.csv', 'baz2a_phd_vs_10mer_8_1_renumbered.csv'],
    ]

    from data import data_info

    algorithms  = ['RA', 'RASmart', 'SD', 'SDSmart']
    #algorithms = ['RA', 'RASmart', 'SD']

    for dataset in data_dundee:
        data_path = data_info.dundee_data_path_V2
        print(dataset)
        file0 = dataset[0]
        file1 = dataset[1]

        free_peaks_file = str(data_path.joinpath(file0))
        with_ligand_peaks_file = str(data_path.joinpath(file1))
        print("|| ", free_peaks_file, with_ligand_peaks_file, " ||")

        for algorithm in algorithms:
            run_assignment(
                free_peaks_file,
                with_ligand_peaks_file,
                out_file=None,
                plot=None,
                algorithm=algorithm)


if __name__ == "__main__":
        main()