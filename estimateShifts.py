"""
Usage:
"""
import argparse
from datetime import datetime
import core.proxy as proxy
import core.graphics as graphics
from core.metrics import custom_accuracy
import sys
import pathlib
from utils.peacks_testannotatedV2 import TestAnnotated as DataParser
import configparser

now = datetime.now().strftime("%Y%m%d%H%M%S")
parser = argparse.ArgumentParser(description="Automatic peaks assignement tool")

# cvs file free protein
parser.add_argument(
    "--free",
    help="csv containing the free protein peaks",
    required=True
)

# csv file protein + ligand
parser.add_argument(
    "--with_ligand",
    help="csv containing the protein+ligand peaks",
    required=True
)

# files are labeled ???
# se si, allora stimare accuracy, else, solo predizioni...
parser.add_argument(
    "--labeled",
    action='store_true',
    help="[optional]: indicates that residues in protein+ligand system are labeled"
    # help="the name of the column (if exists) containing residue label/index",
)

#parser.add_argument(
#    "--label_column",
#    default=None)


parser.add_argument(
    "--out_file",
    default=None)


parser.add_argument(
    "--alg",
    choices=['SD', 'SDSmart', 'RA', 'RASmart'],
    help="The algorithm")


parser.add_argument(
    "--plot",
    default=False,
    action="store_true",
    help="Plotta qualcosa")


def run_assignment(free_protein, with_ligand_protein, out_file, plot, algorithm, labeled):
    print("Free: ", free_protein)
    print("With ligand: ", with_ligand_protein)
    free_data_filename = pathlib.PurePath(free_protein).name.split('.csv')[0]
    withligand_data_filename = pathlib.PurePath(with_ligand_protein).name.split('.csv')[0]
    print("Full Paths: ", free_data_filename, withligand_data_filename)


    # RUN ALGORITHM
    assignemts, f_peaks, wl_peaks, ACC, _ = proxy.estimate_shifts(
        free_protein,
        with_ligand_protein,
        assignment_algorithm=algorithm,
        cerm_data=False,
        labeled=labeled)

    assignemts.index.name = "ResidueKey"

    if out_file != None:
        output_filename = out_file
    else:
        output_filename = free_data_filename+"--"+withligand_data_filename+"--"+algorithm
        # LA SEGUENTE SOLO SE E' ETICHETTATO
        if labeled==True:
            output_filename = output_filename + "({:.2f}).csv".format(ACC * 100)
        else:
            output_filename = output_filename+".csv"


    print("OUTPUT FILE NAME ===> ", output_filename)

    print(assignemts)

    if labeled:
        assignemts = assignemts[['order', 'x', 'y', 'assigned_to', 'pred_x', 'pred_y', 'est_shift', 'real_shift', 'target_x', 'target_y']]

    else:
        assignemts = assignemts[['order', 'x', 'y', 'assigned_to', 'pred_x', 'pred_y', 'est_shift']]

    assignemts.to_csv(output_filename, sep=';')

    # PROCESS RESULTS
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


def multiple_classification(systems):
    for free, with_ligand in systems:
        run_assignment(free, with_ligand)


def main(args):

    #if args.dataset == None:
    #    print("Normale classificazione")
    run_assignment(args.free, args.with_ligand, args.out_file, args.plot, args.alg, args.labeled)

    #else:
    #    print("We are dealing with more systems...")
    #    multiple_classification()



def run_dummy():
    # args = parser.parse_args()
    from data import data_info

    data_dundee = [
        ['Ube2T_ref_final.csv', 'ube2t_em02_5mM_manual.csv'],
        ['Ube2T_ref_final.csv', 'ube2t_em11_3mM_manual.csv']]

    data_cerm = [

        ["MMP12_AHA_ref.txt", "MMP12Cat_NNGH_T1_ref_300707.txt"],
        #["MMP12_AHA_ref.txt", "MMP12Cat_Dive_T1_ref_peaks.txt"],

        #["CAIIZn_000_furo_03_170317.txt", "CAIIZn_100_furo_11_170317.txt"],
        #["CAIIZn_0.00_sulpiride_03_040417.txt", "CAIIZn_5mM_sulpiride_19_040417.txt"],
        #["CAII_Zn_000_pTulpho_03_220317.txt","CAII_Zn_100f_pTulpho_18_220317.txt"],
        #["CAII_Zn_000_pTS_04_291216.txt", "CAII_Zn_100_pTS_15_291216.txt"],
        #["CAII_Zn_000_oxalate_04_221116.txt","CAII_Zn_15mM_oxalate_31_221116.txt"],

        #["CAIIDM_Zn_free_T1_ref_20_081020.txt", "CAII_DM_Zn_SCN_T1_ref_051020.txt"],
        #["CAII_DM_Co_free_onlyAssigned.txt", "CAII_DM_Co_SCN_onlyAssigned.txt"],
    ]

    #data_cerm = [
        # ["CAIIDM_Zn_free_T1_ref_20_081020.txt", "CAII_DM_Zn_SCN_T1_ref_051020.txt"],
        #["CAII_DM_Co_free_onlyAssigned.txt", "CAII_DM_Co_SCN_onlyAssigned.txt"],
    #]

    for dataset in data_cerm:
        data_path = data_info.cerm_data_path

        # for dataset in data_dundee:
        #    data_path = data_info.dundee_data_path_V2
        print(dataset)
        file0 = dataset[0]
        file1 = dataset[1]

        free_peaks_file = str(data_path.joinpath(file0))
        with_ligand_peaks_file = str(data_path.joinpath(file1))

        print("|| ", free_peaks_file, with_ligand_peaks_file, " ||")

        # args = parser.parse_args(['--free', 'system1.csv', '--with_ligand', 'system2.csv', '--out_file', "predictions.csv"])
        args = parser.parse_args(
            ['--free', free_peaks_file, '--with_ligand', with_ligand_peaks_file, "--alg", "SDSmart"])
        print(args)
        main(args)



if __name__ == "__main__":
        args = parser.parse_args()
        print(args)
        main(args)
        #run_dummy()

