#from munkres import Munkres
from ortools.linear_solver import pywraplp
import utils.peacks_testannotatedV2 as pk
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
test = []
test.append(["CAIIZn_000_furo_03_170317","CAIIZn_100_furo_11_170317"])
'''
test.append(["MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510"])
test.append(["MMP12Cat_AHA_T1_ref_270510","MMP12Cat_NNGH_T1_ref_300707"])
test.append(["MMP12Cat_NNGH_T1_ref_300707","MMP12Cat_Dive_T1_ref_peaks"])
test.append(["MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref"])
test.append(["CAIIZn_0.00_sulpiride_03_040417","CAIIZn_5mM_sulpiride_19_040417"])
test.append(["CAII_Zn_000_pTulpho_03_220317","CAII_Zn_100f_pTulpho_18_220317"])
test.append(["CAII_Zn_000_pTS_04_291216","CAII_Zn_100_pTS_15_291216"])
test.append(["CAII_Zn_000_oxalate_04_221116","CAII_Zn_15mM_oxalate_31_221116"])
'''
dataframe = None
datadict = {}
perdict = {}
for p in test:
    ta = pk.TestAnnotated()
    # #f1 ,f2 = ta.getAssignmentData("CAIIZn_000_furo_03_170317","CAIIZn_100_furo_11_170317")
    # f1 ,f2 = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510")
    # f1 ,f2 = ta.getAssignmentData("MMP12Cat_AHA_T1_ref_270510","MMP12Cat_NNGH_T1_ref_300707")
    # f1 ,f2 = ta.getAssignmentData("MMP12Cat_NNGH_T1_ref_300707","MMP12Cat_Dive_T1_ref_peaks")
    # f1 ,f2 = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref")
    # f1 ,f2 = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref")
    # f1, f2 = ta.getAssignmentData("CAIIZn_0.00_sulpiride_03_040417","CAIIZn_5mM_sulpiride_19_040417")
    # f1, f2 = ta.getAssignmentData("CAII_Zn_000_pTulpho_03_220317","CAII_Zn_100f_pTulpho_18_220317")
    # f1, f2 = ta.getAssignmentData("CAII_Zn_000_pTS_04_291216","CAII_Zn_100_pTS_15_291216")
    # #f1, f2 = ta.getAssignmentData("CAII_Zn_000_oxalate_04_221116","CAII_Zn_15mM_oxalate_31_221116")
    f1, f2 = ta.getAssignmentData(p[0],p[1])
    af1 = list(f1.values())
    af2 = list(f2.values())
    nf1 = list(f1.keys())
    nf2 = list(f2.keys())
    # tp= 145
    # f1 = dict(zip(nf1[:tp],af1[:tp]))
    # f2 = dict(zip(nf2[:tp],af2[:tp]))
    #
    #
    # af1 = list(f1.values())
    # af2 = list(f2.values())
    # nf1 = list(f1.keys())
    # nf2 = list(f2.keys())
    if len(list(f1.values())) < len(list(f2.values())) :
        af1 = list(f2.values())
        af2 = list(f1.values())
        nf1 = list(f2.keys())
        nf2 = list(f1.keys())
    s1 = len(af1)
    s2 = len(af2)
    print(s1,s2)
    naf1v = np.array(af1, dtype=float)
    naf2v = np.array(af2, dtype=float)
    print(naf1v.shape,naf2v.shape)
    naf1v[:,1]*= .2
    naf2v[:,1]*= .2
    #naf1 = naf1v[0:tp,:]*100.
    #naf2 = naf2v[0:tp,:]*100.
    naf1 = naf1v
    naf2 = naf2v
    #from munkres import Munkres, print_matrix
    # matrix = [[5, 9, 1],
    #           [10, 3, 2],
    #           [8, 7, 4]]
    matrix = distance_matrix(naf1v,naf2v)
    costs = distance_matrix(naf1v,naf2v )
    cost = costs.T
    names = p[0] + "_" + p[1]
    results = []
    perc = []
    def cascarti(windlist):
        totscarto = 0
        for i in range(len(windlist)-1):
            totscarto += (windlist[i+1] - windlist[i])**2
        return totscarto





     # Otimize N factor
    for va in range(20):
        print("ITERAZIONE ----> ", va)
        #mi = 1/(va+1)
        naf1v = np.array(af1, dtype=float)
        naf2v = np.array(af2, dtype=float)
        print(naf1v.shape,naf2v.shape)
        naf1v[:,1]*= .2
        naf2v[:,1]*= .2
        #matrix = distance_matrix(naf1v,naf2v)
        if va == 0:
            costs = distance_matrix(naf1v,naf2v )
        #print(matrix)
        # using OR-TOLLS
        # https://developers.google.com/optimization/assignment/assignment_example#python_4
        solver = pywraplp.Solver.CreateSolver('assignment_mip', 'CBC')
        num_workers = len(costs)
        num_tasks = len(costs[0])
        # x[i, j] is an array of 0-1 variables, which will be 1
        # if worker i is assigned to task j.
        x = {}
        for i in range(num_workers):
            for j in range(num_tasks):
                x[i, j] = solver.IntVar(0, 1, '')
        # Each worker is assigned to at most 1 task.
        for i in range(num_workers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)
        # Each task is assigned to exactly one worker.
        for j in range(num_tasks):
            solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)
        objective_terms = []
        for i in range(num_workers):
            for j in range(num_tasks):
                objective_terms.append(costs[i][j] * x[i, j])
        solver.Minimize(solver.Sum(objective_terms))
        # Create the mip solver with the CBC backend.
        status = solver.Solve()
        indexes = []
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            #print('Total cost = ', solver.Objective().Value(), '\n')
            for i in range(num_workers):
                for j in range(num_tasks):
                    # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                    if x[i, j].solution_value() > 0.5:
                        #print('Worker %d assigned to task %d.  Cost = %d' %
                        #      (i, j, costs[i][j]))
                        indexes.append((i,j))
        good = 0
        bad = 0
        vdistance =[]
        vdistanceG = []
        vdistancet = []
        hdisti = []
        hdistn = []
        for row, column in indexes:
            #value = matrix[row][column]
            #total += value
            #print('(%d, %d) -> %d' % (row, column, value))
            vdistance.append(costs[row,column])
            vdistancet.append([row,column])
            hdisti.append(row)
            hdistn.append(nf1[row])
            if nf1[row] == nf2[column]:
                good= good+1
                vdistanceG.append(costs[row,column])
            elif nf1[row] in nf2:
                bad = bad +1
                columnG = nf2.index(nf1[row])
                vdistanceG.append(costs[row,columnG])
            else:
                vdistanceG.append(0.0)
        per = good/(bad+good)
        print("Good ", good, " Bad ", bad, " per ", per , "va ", va+1)
        indata = "Good ", good, " Bad ", bad, " % ", per
        results.append(indata)
        perc.append(per)
        #print(vdistance)
        #print(costs)
        scarti = []
        for i in range(len(vdistance)):
            st = []
            for a in range(5):
                a -= 2
                if a+i >= 0 and a+i < len(vdistance):
                   st.append(vdistance[a+i])
            #print("$$$$$$", st)
            #print(cascarti(st))
            scarti.append(cascarti(st))
        meanv = []
        for i in range(len(vdistance)):
            st = []
            #print("new")
            for a in range(5):
                a -= 2
                #print(a, i+a)
                if a + i >= 0 and a + i < len(vdistance):
                #if a+i >= 0 and a+i < len(vdistance) and a != 0:
                    st.append(vdistance[a+i])
            nst = np.array(st)
            #print(nst,nst.mean())
            meanv.append(nst.mean())
        print("vdistance", len(vdistance))

        import sys
        sys.exit()

        #print(vdistancet)
        #for i in range(10):
        im = vdistance.index(max(vdistance))
        print(im,vdistance[im],meanv[im])
        if nf1[im] == nf2[im]:
            print("Good")
        else:
            print("Bad")
        print(costs.shape)
        print(len(meanv))
        indrowlist = [x[0] for x in indexes]
        indcollist = [x[1] for x in indexes]
        print(len(indrowlist), len(indcollist))
        for ic in range(costs.shape[0]):
            #print(ic in indrowlist, ic )
            if ic in indcollist:
                costs[ic,:] *= 1/meanv[ic]
                #costs[ic,:] += scarti[ic]*3
        #if vdistance[im] > meanv[im]*3.:
            #print("Aumento ", i,vdistance[i], meanv[i]*3.)
            #print(costs[vdistancet[i][0],vdistancet[i][1]])
         #   costs[vdistancet[im][0],vdistancet[im][1]] = costs[vdistancet[im][0],vdistancet[im][1]] + meanv[im]/3
            #print(costs[vdistancet[i][0],vdistancet[i][1]])
    datadict[names] = results
    perdict[names] = perc
    print(datadict)
percdataframe = pd.DataFrame(perdict)
dataframe = pd.DataFrame(datadict)
print(dataframe)
dataframe.to_csv("peaks.csv", index = False)
percdataframe.to_csv("perpeaks.csv", index = False)