import sys
from peaksIdentification.peaks_assignement import *
import pickle
import numpy as np
filename = "spect.pck"
infile = open(filename,'rb')
spec_dict = pickle.load(infile)
infile.close()
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
