import numpy as np
import sys
import re
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
# from peaksIdentification.peaks_assignement import generate_data, reassign_peaks
from core.peak_manager_V2 import PeakManager, Spectra


class TestAnnotated:
    def __init__(self, path="./data/Annotated_spectra/"):
        self.path = path
        #fin1 = self.path + files[0] + ".txt"
        #fin2 = self.path + files[1] + ".txt"
        #print("*****************************+", fin1, fin2, "*****************************")
        #self.ass1 = self.__readtxt(fin1)
        #self.ass2 = self.__readtxt(fin2)

    def getAssignmentData(self, filein1, filein2):
        fin1 = self.path + filein1 + ".txt"
        fin2 = self.path + filein2 + ".txt"
        return self.__readtxt(fin1), self.__readtxt(fin2)



    def __readtxt(self, filein):
        parse = open(filein, "r").readlines()
        sparse = filter(lambda x: re.match(r"^H/N[ \t]+\w+[ \t]+\w+[ \t]+[0-9 .]+[ \t]+[0-9 .]+", x), parse)
        mapout = dict(map(lambda x: [re.split(r"[ \t]+", x)[1],
                                     [re.split(r"[ \t]+", x)[3], re.split(r"[ \t]+", x)[4]]], sparse))
        return mapout


    def set_spe1(self,spe1):
        print("setted spe1")
        self.__spe1 = spe1

    def set_spe2(self,spe2):
        self.__spe2 = spe2


    def setSearch_depth(self, search_depth):
        self.search_depth = search_depth

    def normSpect(self, in1, in2):
        if in1.shape[0] > in2.shape[0]:
            gra = in1
            pic = in2
            inv = False
        elif in1.shape[0] < in2.shape[0]:
            gra = in2
            pic = in1
            inv = True
        else:
            return in1, in2

        dim = gra.shape[0]
        diff = gra.shape[0] - pic.shape[0]
        zpeaks = np.zeros((dim, 2))
        zpeaks[:-diff, :] = pic
        if inv:
            return zpeaks, gra
        else:
            return gra, zpeaks

    def findkey(self, dict, valuein):
        return [name for name, value in dict.items() if (valuein[0] == float(value[0][0]) and valuein[1] == float(value[0][1]))]

    def getResult(self):
        return self.results

    # def plot(self):

    def getDataFrame(self):
        if (self.__spe1 is not None) and (self.__spe2 is not None):
            return self.__spe1, self.__spe2


    def readsimple(self):

        for fi in self.list_files:

            fin1 = self.path + fi[0] + ".txt"
            fin2 = self.path + fi[1] + ".txt"
            print("*****************************+", fin1, fin2, "*****************************")
            ass1 = self.readtxt(fin1)
            ass2 = self.readtxt(fin2)
            print(ass1)
            sys.exit()
            npa1 = []
            npa2 = []
            for name, value in ass1.items():
                if name in ass2.keys():
                    npa1.append(value[0])
                    npa2.append(ass2[name][0])

            for name, value in ass1.items():
                if name not in ass2.keys():
                    npa1.append(value[0])

            for name, value in ass2.items():
                if name not in ass1.keys():
                    npa2.append(value[0])

            npa1t = np.array(list(map(lambda x: x[0], ass1.values())))
            npa2t = np.array(list(map(lambda x: x[0], ass2.values())))
            npa1 = np.array(npa1)
            npa2 = np.array(npa2)
            npa1 = np.asfarray(npa1, float)
            npa2 = np.asfarray(npa2, float)
            print("Setting spe1")
            self.set_spe1(npa1)
            self.set_spe2(npa2)

    def read(self):
        resultcomp = {}
        for fi in self.list_files:

            fin1 = self.path + fi[0] + ".txt"
            fin2 = self.path + fi[1] + ".txt"
            print("*****************************+", fin1, fin2, "*****************************")

            ass1 = self.readtxt(fin1) # dictionary [{'aa': (x, y) }, {}]
            ass2 = self.readtxt(fin2) # dictionary

            npa1 = []
            npa2 = []


            for name, value in ass1.items():
                if name in ass2.keys():
                    npa1.append(value[0]) # value[0] e' la tupla
                    npa2.append(ass2[name][0]) # ass2[name][0] e' la  nuova tupla


            for name, value in ass1.items():
                if name not in ass2.keys():
                    npa1.append(value[0])


            for name, value in ass2.items():
                if name not in ass1.keys():
                    npa2.append(value[0])


            npa1t = np.array( list(map(lambda x: x[0], ass1.values())) )
            npa2t = np.array( list(map(lambda x: x[0], ass2.values())) )

            npa1 = np.array(npa1)
            npa2 = np.array(npa2)

            npa1 = np.asfarray(npa1, float)
            npa2 = np.asfarray(npa2, float)
            print("Setting spe1")
            self.set_spe1(npa1)
            self.set_spe2(npa2)
            # np.savetxt("save1.csv",npa1,delimiter=',')
            # for i,a in zip(npa2, ass2.keys()):
            #    print(i,a, ass2[a], findkey(ass2, i))
            # print(npa1)
            # print(npa1t)
            print(npa1.shape, npa1t.shape, npa2.shape, npa2t.shape)
            # v1, v2 = self.normSpect(npa1, npa2)

            # allineo i vettori

            # perc = 0.01
            # peakmanager, score = reassign_peaks(v1, v2, perc, DEBUG=True)
            # X, Y = peakmanager.get_couples()
            old_spectra = Spectra(npa1, suffix='p')
            new_spectra = Spectra(npa2, suffix='s')
            # pm = PeakManager()
            # pm = PeakManager(search_depth=self.search_depth)
            pm = PeakManager(search_depth=4, max_search_per_level=10, log=False)
            X, Y = pm.getAssociations(old_spectra, new_spectra)

            self.x = X
            self.y = Y
            good = 0
            bad = 0
            assfin = []
            s1name = []
            s2name = []
            hdist = []
            hdistg = []
            hdistn = []
            hdisti = []

            ##################################################################

            for x, y in zip(X, Y):
                # estra il nome del aa dal dizionario
                key1 = self.findkey(ass1, x)
                key2 = self.findkey(ass2, y)
                s1name.append(key1[0])
                s2name.append(key2[0])
                # print("PICCHI",x,y)
                # print(np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2))
                distt = np.sqrt( (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 )
                #
                if key1[0] in ass1.keys() and key1[0] in ass2.keys():
                    # print(ass1[key1[0]])
                    # print(ass2[key1[0]])
                    x1 = float( ass1[key1[0]] [0][0] )
                    x2 = float( ass1[key1[0]] [0][1] )
                    y1 = float( ass2[key1[0]] [0][0] )
                    y2 = float( ass2[key1[0]] [0][1] )
                    distp = np.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
                    hdist.append(distt)
                    hdistg.append(distp)
                    hdistn.append(key1[0])
                    strnkey1 = ''.join(char for char in key1[0] if char.isnumeric())
                    hdisti.append(int(strnkey1))

                else:
                    distp = 100.0
                # print("distt ->",distt)
                # print("distp ->",distp)

                if self.findkey(ass1, x) == self.findkey(ass2, y):
                    good = good + 1
                    status = "good"
                else:
                    bad = bad + 1
                    status = "bad"
                # print(x, "\t-->", y, "keyx ", self.findkey(ass1,x), "keyy ", self.findkey(ass2,y))
                assfin.append([self.findkey(ass1, x), self.findkey(ass2, y)])
                resultcomp[key1[0]] = {"key2": key2[0], "staus": status, "distt": distt, "distp": distp}
            ##################################################################



            print("Good = ", good)
            print("Bad = ", bad)
            print("Perc good", good / (bad + good))
            print("deepsearch= ", self.search_depth)
            # results[fi[0]+"_"+fi[1]] = {"Good": good, "Bad": bad, "Perc ": good/(bad+good), "assfin": assfin}
            self.results[fi[0] + "_" + fi[1]] = {"Good": good, "Bad": bad, "Perc ": good / (bad + good),
                                                 "dim": [npa1.shape, npa2.shape], "statiscic": resultcomp}
            fig = go.Figure()
            fig.update_layout(width=1300, height=1000)
            fig['layout']['xaxis']['autorange'] = "reversed"
            fig['layout']['yaxis']['autorange'] = "reversed"
            fig.add_trace(
                go.Scatter(
                    mode='markers+text',
                    x=X[:, 0],
                    y=X[:, 1],
                    marker=dict(
                        color='LightSkyBlue',
                        size=4,
                        line=dict(
                            color='MediumPurple',
                            width=1
                        )
                    ),
                    name='Spectra1',
                    text=s1name, textposition="bottom center"
                ))
            fig.add_trace(
                go.Scatter(
                    mode='markers+text',
                    x=Y[:, 0],
                    y=Y[:, 1],
                    marker=dict(
                        color='Coral',
                        size=4,
                        line=dict(
                            color='MediumPurple',
                            width=1
                        )
                    ),
                    name='Spectra2',
                    text=s2name, textposition="bottom center"
                ))
            for x, y in zip(X, Y):
                key1 = self.findkey(ass1, x)
                key2 = self.findkey(ass2, y)
                if key1[0] == key2[0]:
                    fig.add_trace(go.Scatter(x=[x[0], y[0]], y=[x[1], y[1]], mode='lines', showlegend=False,
                                             line=dict(color="MediumPurple")))
                else:
                    fig.add_trace(go.Scatter(x=[x[0], y[0]], y=[x[1], y[1]], mode='lines', showlegend=False,
                                             line=dict(color="red")))

            fig.show()

            print(hdist)
            print(hdistn)
            print(hdisti)
            fig1 = go.Figure(data=[go.Histogram(x=hdist)])
            fig1.show()
            df1 = pd.DataFrame({"DistanceC": hdist, "DistanceG": hdistg, "Index": hdisti, "Name": hdistn})
            df1 = df1.sort_values(by=['Index'])
            print(df1)
            fig2 = px.bar(df1, x="Name", y="DistanceC")
            fig2.show()
            fig3 = px.bar(df1, x="Name", y="DistanceG")
            fig3.show()

            fig4 = go.Figure()
            fig4.update_layout(width=1300, height=1000)
            fig4.add_trace(go.Bar(
                x=df1["Name"],
                y=df1["DistanceG"],
                name='DistanceG',
                marker_color='blue'
            ))
            fig4.add_trace(go.Bar(
                x=df1["Name"],
                y=df1["DistanceC"],
                name='DistanceC',
                marker_color='red'
            ))

            # Here we modify the tickangle of the xaxis, resulting in rotated labels.
            fig4.update_layout(barmode='group', xaxis_tickangle=-45)
            fig4.show()

            # return self.results
            # print(self.results)


