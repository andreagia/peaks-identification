


import argparse
from datetime import datetime
import core.proxy as proxy
import core.graphics as graphics
from core.metrics import custom_accuracy
import sys
import pathlib
from utils.peacks_testannotatedV2 import TestAnnotated as DataParser
import configparser


def run_assignment(free_protein, with_ligand_protein, out_file, plot, algorithm, labeled):
    print("Free: ", free_protein)
    print("With ligand: ", with_ligand_protein)

    free_data_filename = pathlib.PurePath(free_protein).name.split('.csv')[0]
    withligand_data_filename = pathlib.PurePath(with_ligand_protein).name.split('.csv')[0]

    print("Full Paths: ", free_data_filename, withligand_data_filename)

    if out_file != None:
        output_filename = out_file
    else:
        output_filename = "#"+free_data_filename+"--"+withligand_data_filename+"--"+algorithm

    # In general....
    # GET DATA
    # load_data()

    # RUN ALGORITHM
    assignemts, f_peaks, wl_peaks, ACC, w_half_size = proxy.estimate_shifts(
        free_protein,
        with_ligand_protein,
        assignment_algorithm=algorithm,
        cerm_data=False,
        labeled=labeled) #

    #if w_half_size != None:
    #    output_filename=f"w{w_half_size}"+output_filename

    if out_file != None:
        output_filename = out_file
    else:
        if w_half_size != None:
            output_filename = free_data_filename + "--" + withligand_data_filename + "--" + algorithm + "[w" + str(w_half_size) +"]"
        else:
            output_filename = "#"+free_data_filename+"--"+withligand_data_filename+"--"+algorithm


    # PROCESS RESULTS
    assignemts.index.name="ResidueKey"
    assignemts.to_csv(output_filename+"({:.2f}).====csv".format(ACC*100), sep=';')


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

    # plot()
    if plot == True:
        graphics.plotProfile(assignemts, free_protein, with_ligand_protein, acc=acc)
        graphics.plotPeaksShifts(f_peaks, wl_peaks, assignemts, free_protein, with_ligand_protein, acc=acc)

    print("fine")




def main():
    # args = parser.parse_args()
    from data import data_info


    data_cerm__ = [

        ["MMP12_AHA_ref.txt", "MMP12Cat_NNGH_T1_ref_300707.txt"],
        ["MMP12_AHA_ref.txt", "MMP12Cat_Dive_T1_ref_peaks.txt"],

        ["CAIIZn_000_furo_03_170317.txt", "CAIIZn_100_furo_11_170317.txt"],
        ["CAIIZn_0.00_sulpiride_03_040417.txt", "CAIIZn_5mM_sulpiride_19_040417.txt"],
        ["CAII_Zn_000_pTulpho_03_220317.txt","CAII_Zn_100f_pTulpho_18_220317.txt"],
        ["CAII_Zn_000_pTS_04_291216.txt", "CAII_Zn_100_pTS_15_291216.txt"],
        ["CAII_Zn_000_oxalate_04_221116.txt","CAII_Zn_15mM_oxalate_31_221116.txt"],

        ["CAIIDM_Zn_free_T1_ref_20_081020.txt", "CAII_DM_Zn_SCN_T1_ref_051020.txt"],
        ["CAII_DM_Co_free_onlyAssigned.txt", "CAII_DM_Co_SCN_onlyAssigned.txt"],
    ]


    data_cerm = [

        #["MMP12_AHA_ref.csv", "MMP12Cat_NNGH_T1_ref_300707.csv"],
        #["MMP12_AHA_ref.csv", "MMP12Cat_Dive_T1_ref_peaks.csv"],
        ["MMP12Cat_AHA_T1_ref_270510.csv", "MMP12Cat_NNGH_T1_ref_300707.csv"],
        ["MMP12Cat_AHA_T1_ref_270510.csv", "MMP12Cat_Dive_T1_ref_peaks.csv"],

        ["CAIIZn_000_furo_03_170317.csv", "CAIIZn_100_furo_11_170317.csv"],
        ["CAIIZn_0.00_sulpiride_03_040417.csv", "CAIIZn_5mM_sulpiride_19_040417.csv"],
        ["CAII_Zn_000_pTulpho_03_220317.csv","CAII_Zn_100f_pTulpho_18_220317.csv"],
        ["CAII_Zn_000_pTS_04_291216.csv", "CAII_Zn_100_pTS_15_291216.csv"],
        ["CAII_Zn_000_oxalate_04_221116.csv","CAII_Zn_15mM_oxalate_31_221116.csv"],

        ["CAIIDM_Zn_free_T1_ref_20_081020.csv", "CAII_DM_Zn_SCN_T1_ref_051020.csv"],
        ["CAII_DM_Co_free_onlyAssigned.csv", "CAII_DM_Co_SCN_onlyAssigned.csv"],
    ]


    data_dundee = [
        #['Ube2T_ref_final.csv', 'ube2t_em02_5mM_manual.csv'],
        ['Ube2T_ref_final.csv', 'ube2T_em11_3mM_manual.csv'],
        #['Ube2T_ref_final.csv', 'ube2t_em04_3mM_manual.csv'],
        #['Ube2T_ref_final.csv', 'ube2t_em09_3mM_manual.csv'],
        #['Ube2T_ref_final.csv', 'ube2t_em17_3mM_final.csv'],
        #['Ube2T_ref_final.csv', 'ube2t_em29_3mM_manual.csv'],

        #['baz2b_phd_ref_renumbered.csv', 'baz2b_vs_5-mer_20_1_renumbered.csv'],
        #['baz2a_phd_ref_renumbered.csv', 'baz2a_phd_vs_5mer_64_1_renumbered.csv'],
        #['baz2a_phd_ref_renumbered.csv', 'baz2a_phd_vs_10mer_8_1_renumbered.csv'],
    ]

    demo_data = [['free.csv', 'with_ligand.csv']]

    #algorithms  = ['RA', 'RASmart', 'SD', 'SDSmart']
    algorithms = ['RASmart']

    for dataset in data_dundee:
    #for dataset in demo_data:
        #data_path = data_info.cerm_csv_data_path
        data_path = data_info.dundee_data_path_V2
        #data_path =pathlib.Path('.')
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
                algorithm=algorithm,
                labeled=True)


if __name__ == "__main__":

        #main(args)
        main()