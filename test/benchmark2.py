"""
Testing performances of the module
"""

import sys
from peaksIdentification.peaks_assignement import *
import pickle
import numpy as np
filename = "spect.pck"
infile = open(filename,'rb')
spec_dict = pickle.load(infile)
infile.close()
protein_length = [100, 200, 500, 1000, 2000]

shift_perc = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3] #

n_shifts = [5, 10, 15, 20, 25, 30, 40, 50]

""" peaks, new_peaks = generate_data( n_peaks=100, n_shifts=10)
print(new_peaks)
print(type(peaks))
sys.exit() """
for sp in spec_dict.keys():
    print(spec_dict[sp].keys())
    numa = [int(i) for i in spec_dict[sp].keys()]
    numa.sort()
    peaks = np.array(spec_dict[sp][str(numa[0])])
    new_peaks = np.array(spec_dict[sp][str(numa[1])])
    if new_peaks.size < peaks.size:
        peaks = np.array(spec_dict[sp][str(numa[1])])
        new_peaks = np.array(spec_dict[sp][str(numa[0])])
    print(peaks.shape,new_peaks.shape)
    print(peaks.size,new_peaks.size)
    N = int(new_peaks.size/2)
    ofN = int(N - peaks.size/2)
    print(N,ofN)
    zpeaks = np.zeros((2,N))
    print(zpeaks.size)
    zpeaks[:,:-ofN] = peaks
    print(zpeaks)
    peaks=zpeaks
    print(peaks.shape,new_peaks.shape)
    
    shift_perc=float(peaks.size/2)/(new_peaks.size-peaks.size/2)
    print("SHIFT PERC")
    print(shift_perc)
    peaks = peaks.T
    new_peaks = new_peaks.T
    score = reassign_peaks(peaks, new_peaks, shift_perc)
    print("-<<<<<", score)
    sys.exit()
    peaks1, new_peaks1 = generate_data( n_peaks=20, n_shifts=5)
    shift_perc1 = (float(20) / 5)
    print("peacks1",peaks1, peaks1.size)
    print("new_peacks1",new_peaks1, new_peaks1.size)
    sys.exit()
    print(peaks)
    print(type(peaks1))
    print(type(peaks))
    print(float(peaks.size))
    print(new_peaks.size-peaks.size)
    print("size peaks1", peaks1.size)
    print("size new_peacks1", new_peaks1.size)
    score1 = reassign_peaks(peaks1, new_peaks1, shift_perc1)

    print("Score1", score1)
    sys.exit()
    score = reassign_peaks(peaks, new_peaks, shift_perc) 
    print("Score", score)
    sys.exit()
sys.exit()
for ns in n_shifts:
    acc_list = []
    print("==" * 40)
    for L in protein_length:
        print("--"*40)
        print("# of peaks: ", L, ", # of shifted peaks", ns)

        stats = []
        for run in range(10):
            shift_perc = (float(ns) / L)
            print("SHIFT PERC: ", shift_perc)

            peaks, new_peaks = generate_data( n_peaks=L, n_shifts=ns)
            print(new_peaks)
            print(peaks)
            print(type(peaks))
            sys.exit()
            #print(type(peaks), peaks.shape)
            #print(type(new_peaks), new_peaks.shape)

            _, score = reassign_peaks(peaks, new_peaks, shift_perc)

            #print(score)

            stats.append(score)
        stats = np.array(stats)
        acc_list.append(stats)
        print("*** SCORE (%): ", stats.mean(), "+/-", stats.std(), "***")

    fig7, ax7 = plt.subplots()
    plt.title("# of shifts: {}".format(ns))
    #plt.title("# of peaks: ", L)
    plt.ylim(0, 100)

    print(shift_perc, L)

    plt.boxplot(acc_list, labels = protein_length,  showfliers=False)
    plt.xlabel("# tot. of peaks")
    plt.ylabel("Peaks assignment accuracy")
    #plt.title("# of peaks: ", L)
    plt.show()


