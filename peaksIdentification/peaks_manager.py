
"""
Algoritmo per la identificazione dei picchi

Steps:
1) Crescita del raggio dei picchi che sono senza il 'new peak'. Stop quando viene identificato il 'new peak'
a - Definizione iniziale del raggio
b - incremento e valutazione di stop (picco per picco)

2) Risoluzione dei casi irrisolti


"""
import numpy as np

def initial_radius(X):
    assert len(X.shape) == 2 and X.shape[1] > 1
    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))
    #print(dist)
    #print(X.shape)
    distances = dist[np.triu_indices(X.shape[0], k=1)]
    #print(distances)
    return distances.mean()

def distance_matrix(X, Y):
    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1))
    assert dist.shape[0] == X.shape[0] and dist.shape[1] == Y.shape[0]
    return dist



class PeakManager:

    def __init__(self, orig_peaks, new_peaks):
        #assert orig_peaks.shape == new_peaks.shape
        self.orig_peaks = orig_peaks
        self.new_peaks = new_peaks
        self.distance = self.distance_matrix(orig_peaks, new_peaks)
        self.radius = -1
        # self.peaks_radius = np.ones(orig_peaks.shape[0])
        self.peaks_radius = None
        self.newPeaksFree = np.ones(orig_peaks.shape[0], dtype = bool)
        self.origPeaksToFix = np.ones(new_peaks.shape[0], dtype = bool)
        self.couples = {}
        # print("Peak Manager Created.")

    def distance_matrix(self, X, Y):
        print("Size X Y")
        print(X.size,Y.size)
        dist = np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1))
        assert dist.shape[0] == X.shape[0] and dist.shape[1] == Y.shape[0]
        return dist


    # get free peaks()
    def get_new_peak_to_fix(self):
        """ Indentifica i NUOVI picchi non ancora assegnati"""
        # calcola il minimo sulle singole colonne
        # i new_peaks per cui questi minimi sono MAGGIORI di un certo valore,
        # sono da fixare/assegnare.
        new_peaks_to_fix = (self.distance.min(axis=0) > self.peaks_radius)
        #print(new_peaks_to_fix)
        #new_peaks_to_fix = new_peaks_to_fix*self.newPeaksFree
        #print( new_peaks_to_fix)
        return new_peaks_to_fix

    def calculate_peaks_status(self):
        #Indentifica i picchi ORIGINALI che non hanno ancora un nuovo picco assegnato

        '''
        TODO: Attenzione, i new picchi che sono stati gia' assegnati
        non vanno guardati quando si cercano quelli da assegnare piu' vicini!!!!

        TODO: Se capitano due picchi nel raggio ????

        TODO: Dobbiamo in qualche modo marcare le coppie di picchi gia assegnate.
        '''
        rows = self.origPeaksToFix
        cols = self.newPeaksFree

        # estrae la sottomatrice dei picchi ancora non assegnati
        dd = self.distance[[rows]][:,cols]

        # gli indici dei minimi
        min_idx = dd.argmin(axis=1) # indici dei new_peak individuati come "da assegnare"
        min_values = dd[range(len(dd)), min_idx]

        just_fixed_orig = min_values <= self.peaks_radius[rows] # orig_peaks che hanno un new_peak nel raggio
        just_fixed_new = min_idx[just_fixed_orig]

        just_fixed_orig = just_fixed_orig.nonzero()[0] # bool, quindi viene convertito

        # print("> Fixed orig: ",  just_fixed_orig, len(just_fixed_orig) )
        # print("> Fixed new",  just_fixed_new, len(just_fixed_new))


        # orig_peaks_fixed = (self.distance[rows,cols].min(axis=1) <= self.peaks_radius)

        # setta i picchi come assegnati, ovver "da non fixare"
        #print(self.origPeaksToFix.nonzero())
        #print(self.origPeaksToFix)

        #print("---------------------------------------")
        optf = self.origPeaksToFix.nonzero()[0]
        nptf = self.newPeaksFree.nonzero()[0]

        orig_peaks_fixed_idx = optf[just_fixed_orig]
        new_peaks_fixed_idx = nptf[just_fixed_new]
        # print("> orig_peaks_fixed_idx", orig_peaks_fixed_idx)
        # print("> new_peaks_fixed_idx", new_peaks_fixed_idx)
        for x, y in zip(orig_peaks_fixed_idx, new_peaks_fixed_idx):
            self.couples[x] = y

        #print(optf)
        #print(orig_peaks_fixed_idx)
        self.origPeaksToFix[orig_peaks_fixed_idx] = False
        self.newPeaksFree[new_peaks_fixed_idx] = False
        #print("---------------------------------------")

        #print(self.origPeaksToFix.nonzero())
        #print(self.origPeaksToFix)


    def update_radius(self, step):
        #to_fix = self.get_orig_peak_to_fix()
        to_fix = self.origPeaksToFix
        self.peaks_radius[to_fix ] += step

    def get_couples_old(self):
        orig_peaks = (self.distance.min(axis=1) <= self.peaks_radius) # boolean

        orig_peaks_indeces = orig_peaks.nonzero()

        new_peaks_indeces  = self.distance.argmin(axis=1)

        #print("arg min axis 1: ", new_peaks_indeces)
        #print("Orig index: ", orig_peaks_indeces)
        #print("New index: ", new_peaks_indeces[orig_peaks_indeces])
        new_peaks_indeces = new_peaks_indeces[orig_peaks_indeces]

        return self.orig_peaks[orig_peaks_indeces], self.new_peaks[new_peaks_indeces]

    def get_couples(self):
        "Ritorna le associazioni picchi vecchi con picchi nuovi"

        orig_idxs = list(self.couples.keys())
        new_idxs = list(self.couples.values())

        # print("orig_idxs: ", orig_idxs)
        # print("new_idxs: ", new_idxs)

        #print(self.orig_peaks)

        return self.orig_peaks[orig_idxs], self.new_peaks[new_idxs]


    def status(self):
        print("----------------------------------------------")
        print("STATUS")
        print("Original peaks free: \t", self.origPeaksToFix.astype(int).nonzero())
        #print(self.origPeaksToFix)
        print("New peaks free:\t\t\t", self.newPeaksFree.astype(int).nonzero())
        #print(self.newPeaksFree)

    def score(self, real_shift):
        orig_idxs = np.array(list(self.couples.keys()))
        new_idxs = np.array(list(self.couples.values()))

        #print("orig_idxs", orig_idxs, self.couples.keys())
        #print("new_idxs", new_idxs)
        #print(type(orig_idxs), orig_idxs.shape)
        #print(type(new_idxs), new_idxs.shape)

        wrong = np.sum(orig_idxs != new_idxs)
        #print(orig_idxs != new_idxs)

        error_perc = wrong/real_shift
        acc_perc = 1 -error_perc

        # print("Wrongly re-assigned : ", wrong, ", error (%): ",error_perc*100)
        # print("Accuracy (%): ", acc_perc*100)

        return  acc_perc*100