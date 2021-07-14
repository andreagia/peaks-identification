import numpy as np

def distance_matrix(X, Y, real_dist = False): # list of peaks
    #X = np.array([peak.coordinates for peak in old_peaks])
    #Y = np.array([peak.coordinates for peak in new_peaks])
    #print("///////")

    xx = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    if real_dist is False:
        xx[:, :, 1] *= 0.2

    dist = np.sqrt(np.sum((  xx  )**2, axis=-1))

    assert dist.shape[0] == X.shape[0] and dist.shape[1] == Y.shape[0]
    max_dist = dist.max()
    #dist = dist**2

    return dist

X = np.array([
    [1, 2.],
    [3, 4],
    [5,6]])

Y = np.array([
    [1, 2],
    [3, 4.],
    [5,6]])

print(distance_matrix(X, Y))

dist = np.zeros((len(X), len(Y)))
for r in range(len(X)):
    for c in range(len(Y)):
        print("-----")
        p1 = X[r]
        p2 = Y[c]
        print(p1)
        print(p2)
        diff = (p1 - p2)
        print(diff)
        diff[1]/=5
        print(diff)
        diff = diff**2
        print(diff)
        out = np.sqrt(np.sum(diff))
        print(np.sum(diff))
        print(np.sqrt(np.sum(diff)))
        dist[r,c] = out
print(dist)




