from ortools.linear_solver import pywraplp
import utils.peacks_testannotatedV2 as pk
import numpy as np
from scipy.spatial import distance_matrix

ta = pk.TestAnnotated()
#f1 ,f2 = ta.getAssignmentData("CAIIZn_000_furo_03_170317","CAIIZn_100_furo_11_170317")
f1 ,f2 = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510")
f1 ,f2 = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref")
#list_files=["MMP12Cat_Dive_T1_ref_peaks","MMP12_AHA_ref"] # 2 test

#f1, f2 = ta.getAssignmentData("CAIIZn_0.00_sulpiride_03_040417","CAIIZn_5mM_sulpiride_19_040417")
#f1, f2 = ta.getAssignmentData("CAII_Zn_000_pTulpho_03_220317","CAII_Zn_100f_pTulpho_18_220317")
# f1, f2 = ta.getAssignmentData("CAII_Zn_000_pTS_04_291216","CAII_Zn_100_pTS_15_291216")
#f1, f2 = ta.getAssignmentData("CAII_Zn_000_oxalate_04_221116","CAII_Zn_15mM_oxalate_31_221116")

peaks, new_peaks = ta.getAssignmentData("MMP12Cat_Dive_T1_ref_peaks","MMP12Cat_AHA_T1_ref_270510")

af1 = list(f1.values())
af2 = list(f2.values())
nf1 = list(f1.keys())
nf2 = list(f2.keys())

if len(list(f1.values())) < len(list(f2.values())) :
    af1 = list(f2.values())
    af2 = list(f1.values())
    nf1 = list(f2.keys())
    nf2 = list(f1.keys())

naf1v = np.array(af1, dtype=float)
naf2v = np.array(af2, dtype=float)
print(naf1v.shape,naf2v.shape)

naf1v[:,1]*= .2
naf2v[:,1]*= .2

matrix = distance_matrix(naf1v,naf2v)
costs = distance_matrix(naf1v,naf2v )

print("=====>", costs.shape)
#costs = costs.T
print("=====>", costs.shape)

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
    print('Total cost = ', solver.Objective().Value(), '\n')
    for i in range(num_workers):
        for j in range(num_tasks):
            # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
            if x[i, j].solution_value() > 0.5:
                print('Worker %d assigned to task %d.  Cost = %d' %
                      (i, j, costs[i][j]))
                indexes.append((i,j))

print("==>", indexes)
good = 0.
bad = 0.

for i,j  in indexes:
    if nf1[i] == nf2[j]:
        good+=1
    else:
        bad+=1

per = good/(bad+good)
print("Good ", good, " Bad ", bad, " per ", per )
#fig.show()