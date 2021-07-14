"""
How to use the peaks-assignment tool
"""

import argparse

from peaksIdentification.peaks_assignement import generate_data, reassign_peaks

from datetime import datetime


def set_arguments():
    parser__ = argparse.ArgumentParser(description="Automatic peaks assignement tool")

    # cvs file free protein
    parser__.add_argument("--free", help="csv containing the free protein peaks $$$$$", required=True)

    # csv file protein + ligand
    parser__.add_argument("--with_ligand", help="csv containing the protein+ligand peaks", required=True)

    # files are labeled ???
    # se si, allora stimare accuracy, else, solo predizioni...
    parser__.add_argument(
        "--labeled",
        action='store_true',
        help="[optional]: indicates that residues in protein+ligand system are labeled"
        # help="the name of the column (if exists) containing residue label/index",
    )

    parser__.add_argument("--label_column", default=None)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    parser__.add_argument("--out_file", default="estimated_shifts_{}.csv".format(now))

    return parser__



def main(args):

    N_PEAKS = 20
    N_SHIFTS = 5

    peaks, new_peaks = generate_data( n_peaks=N_PEAKS, n_shifts=N_SHIFTS)
    perc = N_SHIFTS/N_PEAKS

    peakmanager, score = reassign_peaks(peaks, new_peaks, perc, DEBUG=True)

    print("shift perc:", perc)
    print(">> score {} %".format(score))

    X, Y = peakmanager.get_couples()
    for x, y in zip(X, Y):
        print(x, "\t-->", y)


if __name__ == "__main__":
    myparser = set_arguments()
    #args = parser.parse_args()
    args = myparser.parse_args(['--free', 'system1.csv', '--with_ligand', 'system2.csv', '--out_file', "predictions.csv"])
    print("This is main")
    #main(args)
    print(args)


