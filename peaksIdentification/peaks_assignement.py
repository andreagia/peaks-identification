"""
- genera dei picchi
- genera gli spostamenti
- riassegna i picchi
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from peaksIdentification.peaksAndShiftGenerator import *
from peaksIdentification.peaks_manager import *
from pylab import *


def print_peaks(peaks, new_peaks, peaks_radius, to_fix_peaks, peak_manager = None):
    peaks_x = peaks[:, 0]
    peaks_y = peaks[:, 1]
    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(peaks_x, peaks_y, '*')

    # tf sta per to-fix
    for (x,y), tf, rad in zip(peaks, to_fix_peaks, peaks_radius):
        if tf:
            circle = plt.Circle((x,y), rad, color='g', fill=False)
            plt.gcf().gca().add_artist(circle)

    plt.plot(new_peaks[:, 0], new_peaks[:, 1], '.' )
    plt.xlim(5, 12)
    plt.ylim(90, 140)
    if peak_manager is not None:
        X, Y = peak_manager.get_couples()
        # print(X, Y)

        for x, y in zip(X, Y):
            print("------------------------->", x, y)
            if x[0] > 1 and y[0] > 1:
                plt.plot([x[0], y[0]], [x[1], y[1]], 'r-', alpha=0.4)

    plt.show()


def generate_data(n_peaks, percentage_shift=None, n_shifts=None):

    assert (percentage_shift is None) != (n_shifts is None)

    N_PEAKS = n_peaks
    # genera dei punti random
    peaks = generate_peaks(n_peaks=N_PEAKS)
    # genera degli spostamenti random
    shifts = generate_shifts(n_peaks=N_PEAKS)

    if percentage_shift is not None:
        # maskera l'80% degli shift
        # replacement= False is needed
        not_shifted_perc = 1. - percentage_shift
        not_shifted = int(not_shifted_perc * N_PEAKS)
    else:
        not_shifted = N_PEAKS - n_shifts

    #rand_ind = np.random.choice(N_PEAKS, int(not_shifted_perc*N_PEAKS), replace=False)
    rand_ind = np.random.choice(N_PEAKS, not_shifted, replace=False)
    shifts[:, rand_ind] = 0.
    new_peaks = peaks + shifts
    peaks = peaks.T
    new_peaks = new_peaks.T
    return peaks, new_peaks

#print(peaks)
#print(new_peaks)

def reassign_peaks(peaks, new_peaks, shift_perc, DEBUG = False):
    init_rad = initial_radius(peaks) / 50
    rad_step = init_rad
    print("Rad_step", rad_step)
    peakManager = PeakManager(peaks, new_peaks)
    peakManager.radius = init_rad

    distance = distance_matrix(peaks, new_peaks)
    # print(orig_new_dist.min(axis=1))
    peaks_radius = np.ones(peaks.shape[0]) * init_rad
    peakManager.peaks_radius = np.ones(peaks.shape[0]) * init_rad

    # CERCHIAMO I PICCHI DA SISTEMARE
    peakManager.calculate_peaks_status()
    origPeaksToFix = peakManager.origPeaksToFix
    newPeaksFree = peakManager.newPeaksFree
    if DEBUG:
        # li stampiamo
        print_peaks(peaks, new_peaks, peaks_radius, origPeaksToFix)
    origPeaksToFix0 = peakManager.origPeaksToFix.copy()

    # print("ITERATION STEPS:")
    for step in range(30):
        # print(step, ")", "==" * 40)
        # _______________RADIUS UPDATE______________
        # allarghiamo il raggio per chi e' necessario
        init_rad += rad_step
        peakManager.radius += rad_step
        peakManager.update_radius(rad_step)
        # _____________AGGIORNA LO STATO DEI PICCHI_______________
        peakManager.calculate_peaks_status()
        # _____________RICONTROLLIAMO LA SITUAZIONE_______________
        if DEBUG:
            peakManager.status()

        if peakManager.origPeaksToFix.sum() == 0 or step == 29:
            # print("SSSTTTTOOOOOPPPPPP")
            # for couple in peakManager.couples:
            # print(peakManager.couples)
            if DEBUG:
                print_peaks(peaks, new_peaks, peakManager.peaks_radius, origPeaksToFix0, peakManager)

            return peakManager, peakManager.score(shift_perc * len(peaks))


def run():
    N_PEAKS = 1000
    perc_shift = 0.01
    peaks, new_peaks = generate_data(N_PEAKS, perc_shift)
    score = reassign_peaks(peaks, new_peaks, perc_shift)




#run()


'''
sys.exit()
########################################################
N_PEAKS = 500

peaks, new_peaks = generate_data(N_PEAKS, 0.1)

init_rad = initial_radius(peaks)/50
rad_step = init_rad

peakManager = PeakManager(peaks, new_peaks)
peakManager.radius = init_rad

print(">>> init rad", init_rad)

distance = distance_matrix(peaks, new_peaks)
#print(distance)
print(">>> init rad", init_rad)

# print(orig_new_dist.min(axis=1))
peaks_radius = np.ones(peaks.shape[0])*init_rad
peakManager.peaks_radius = np.ones(peaks.shape[0])*init_rad

###########################################################
# CERCHIAMO I PICCHI DA SISTEMARE
peakManager.calculate_peaks_status()

origPeaksToFix = peakManager.origPeaksToFix
newPeaksFree = peakManager.newPeaksFree

# li stampiamo
print_peaks(peaks, new_peaks, peaks_radius, origPeaksToFix)

origPeaksToFix0 = peakManager.origPeaksToFix.copy()

print("ITERATION STEPS:")
for step in range(30):
    print(step,")","=="*40)
    # _______________RADIUS UPDATE______________
    # allarghiamo il raggio per chi e' necessario
    init_rad += rad_step
    peakManager.radius += rad_step
    peakManager.update_radius(rad_step)

    # _____________AGGIORNA LO STATO DEI PICCHI_______________
    peakManager.calculate_peaks_status()

    # _____________RICONTROLLIAMO LA SITUAZIONE_______________
    peakManager.status()

    if peakManager.origPeaksToFix.sum() == 0 or step==29:
        print("SSSTTTTOOOOOPPPPPP")

        #for couple in peakManager.couples:
        #print(peakManager.couples)
        print_peaks(peaks, new_peaks, peakManager.peaks_radius, origPeaksToFix0, peakManager)

        print("SCORE:", peakManager.score(0.2*N_PEAKS))
        break
'''
