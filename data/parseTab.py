import os
import numpy as np
import pickle

dizout = {}

def getxy(str):
    x = []
    y = []
    for i in str:
        if len(i) > 10:
            #print("----"+i+"------")
            if i.split()[0].isdigit():
                x.append(float(i.split()[5]))
                y.append(float(i.split()[6]))
    return np.array([x, y])

ini = ""
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".tab"):
            print(os.path.join(root, file))
            fr = open(os.path.join(root, file), "r").readlines()
            csxy = getxy(fr)
            print(type(csxy))
            print(csxy.shape)
            if root.split(os.sep)[-2] != ini :
                dizout[root.split(os.sep)[-2]] = {}
                ini = root.split(os.sep)[-2]
            #print(dizout)
            dizout[root.split(os.sep)[-2]][root.split(os.sep)[-1]] = csxy
            #print(root.split(os.sep))
            print(root.split(os.sep)[-2])
            print(root.split(os.sep)[-1])
            print(dizout[root.split(os.sep)[-2]].keys())

#print(dizout)
#print(dizout.keys())
filename = 'spect.pck'
outfile = open(filename,'wb')
pickle.dump(dizout,outfile)
outfile.close()