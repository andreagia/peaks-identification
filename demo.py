"""
How to use the peaks-assignment tool
"""

from peaksIdentification.peaks_assignement import generate_data, reassign_peaks

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

