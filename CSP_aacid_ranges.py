import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def clean_data(csvdata):
    for col in csvdata.columns:
        print(">>>", col)
        before = csvdata[col]
        csvdata[col] = csvdata[col].astype(str).replace(['--', "--"], None)
        after = csvdata[col]

        if '--' in csvdata[col].values:
            after_list = after.to_list()
            for j, item in enumerate(after_list):
                if item.encode() == '--'.encode():
                    after_list[j] = None
            csvdata[col] = after_list

    csvdata['Residue Number'] = csvdata['Residue Number'].astype(int)
    csvdata['CSP (ppm)'] = csvdata['CSP (ppm)'].astype(float)
    csvdata['Automated'] = csvdata['Automated'].astype(float)
    #y_exp = csvdata['CSP (ppm)'].astype(float)
    # x2 = csvdata['Residue Number'].astype(int)
    #y_auto = csvdata['Automated'].astype(float)
    # return x, y_exp, y_auto
    return csvdata

def list_for_pymol(lista):
    stringa = ""
    for j, item in enumerate(lista):
        if j != len(lista)-1:
            stringa+=str(item)+"+"
        else:
            stringa += str(item)
    return stringa

def get_high_low_CSPs(csvdata, col):
    U = csvdata[col].mean()
    S = csvdata[col].std()
    Len = len(csvdata)

    print("===", col.upper(), "===")
    print("1 SIGMA:", len((csvdata[
        (csvdata[col].astype(float) < U + S) &
        (csvdata[col].astype(float) > U - S)])) / Len)

    print("2 SIGMA:", len((csvdata[
        (csvdata[col].astype(float) < U + 2 * S) &
        (csvdata[col].astype(float) > U - 2 * S)])) / Len)

    print("3 SIGMA:", len((csvdata[
        (csvdata[col].astype(float) < U + 3 * S) &
        (csvdata[col].astype(float) > U - 3 * S)])) / Len)

    low_CSPs = csvdata[
        (csvdata[col].astype(float) > U + S) &
        (csvdata[col].astype(float) < U + 2 * S)]

    high_CSPs = csvdata[
        csvdata[col].astype(float) > U + 2 * S]

    print("low_CSPs:", len((csvdata[(csvdata['Automated'].astype(float) >= U + S) &
                                    (csvdata['Automated'].astype(float) < U + 2 * S)])))

    print("high_CSPs:", len((csvdata[(csvdata['Automated'].astype(float) >= U + 2 * S)])))

    print(low_CSPs['Residue Number'].astype(int).to_list())
    print(high_CSPs['Residue Number'].astype(int).to_list())

    high_CSPs = high_CSPs['Residue Number'].astype(int).to_list()
    low_CSPs = low_CSPs['Residue Number'].astype(int).to_list()

    high_CSPs = list_for_pymol(high_CSPs)
    low_CSPs = list_for_pymol(low_CSPs)

    return high_CSPs, low_CSPs, U, S


print(os.getcwd())
path = "/Users/vincenzo/CERM/PeaksIdentification/testDundee110221_conversed"
files = os.listdir(path)
print(files)
plot_path = os.path.join(path, "CSP_two_ranges")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

for file in files:
    if file.endswith(".csv"):
        prot, with_lig, alg_ = file.split("--")
        alg_ = alg_[:alg_.index(".csv")]

        ii = alg_.index('(')
        alg = alg_[:ii]


        if alg=="RASmart":
            print("\n", "///" * 20)
            print(file)
            print(alg)
            csvdata = pd.read_csv(os.path.join(path, file))
            csvdata = clean_data(csvdata)

            high_csp_auto, low_csp_auto, u_auto, s_auto = get_high_low_CSPs(csvdata, "Automated")
            high_csp_exp, low_csp_exp, u_exp, s_exp = get_high_low_CSPs(csvdata, "CSP (ppm)")
            file_path = os.path.join(plot_path, f"{prot}--{with_lig}[{alg_}].txt")

            print("FILE :", file_path)
            with open(file_path, 'w') as f:
                f.writelines("high_csp_EXP" + "\n")
                f.writelines(high_csp_exp+"\n\n")
                f.writelines("low_csp_EXP"+"\n")
                f.writelines(low_csp_exp + "\n\n")

                f.writelines("\n"+str(u_exp)+" "+ str(s_exp)+"\n\n")

                f.writelines("high_csp_AUTO" + "\n")
                f.writelines(high_csp_auto+"\n\n")
                f.writelines("low_csp_AUTO"+"\n")
                f.writelines(low_csp_auto + "\n")

                f.writelines("\n"+str(u_auto)+", "+str(s_auto)+"\n")

        #sys.exit()