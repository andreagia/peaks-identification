
import sys
import os
import pandas as pd
pd.set_option('display.max_rows', 1000)
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

    return csvdata

def list_for_pymol(lista):
    stringa = ""
    for j, item in enumerate(lista):
        if j != len(lista)-1:
            stringa+=str(item)+"+"
        else:
            stringa += str(item)
    return stringa

def get_high_low_CSPs(csvdata, col, auto=False):
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

    if auto is False:
        # EXPERIMENTAL DATA
        high_CSPs = csvdata[csvdata[col].astype(float) > U + 2 * S]
    else:
        # PREDICTED DATA
        # identify CSPs > 3*sigma
        gt3sigma_idxs = (csvdata[col].astype(float) > U + 4*S).to_numpy().nonzero()[0]

        # remove those from the data
        lt3sigma_idxs = [idx for idx in csvdata.index.values if idx not in gt3sigma_idxs]
        #print(gt3sigma_idxs.to_numpy().astype(int))
        #print(gt3sigma_idxs.to_numpy().nonzero()[0])
        #print(csvdata.iloc[gt3sigma_idxs.to_numpy().nonzero()[0]])
        print(lt3sigma_idxs)
        print(gt3sigma_idxs)
        print("tot len: ", len(set(csvdata.index.values)))
        print("gt3s len: ", len(set(gt3sigma_idxs)))
        print("lt3s len: ", len(set(lt3sigma_idxs)))

        csvdata_senza = csvdata.iloc[lt3sigma_idxs]

        # recalculate \mu and \sigma after removing CSPs > \mu + 3*\sigma
        U = csvdata_senza[col].mean()
        S = csvdata_senza[col].std()

        high_CSPs = csvdata[
            (csvdata[col].astype(float) > U + 2 * S) #&
            #(csvdata[col].astype(float) < U + 3 * S)
            ]

        low_CSPs = csvdata[
            (csvdata[col].astype(float) > U + S) &
            (csvdata[col].astype(float) < U + 2 * S)]

    #print("low_CSPs:", len((csvdata[(csvdata['Automated'].astype(float) >= U + S) &
    #                                (csvdata['Automated'].astype(float) < U + 2 * S)])))

    #print("high_CSPs:", len((csvdata[(csvdata['Automated'].astype(float) >= U + 2 * S)])))

    #print(low_CSPs['Residue Number'].astype(int).to_list())
    #print(high_CSPs['Residue Number'].astype(int).to_list())

    high_CSPs = high_CSPs['Residue Number'].astype(int).to_list()
    low_CSPs = low_CSPs['Residue Number'].astype(int).to_list()

    high_CSPs = list_for_pymol(high_CSPs)
    low_CSPs = list_for_pymol(low_CSPs)

    return high_CSPs, low_CSPs, U, S


print(os.getcwd())
path = "/Users/vincenzo/CERM/PeaksIdentification/testCERM110221_conversed"
macro_path = os.path.join(path, "CSP_macros20210610")

if not os.path.exists(macro_path):
    os.makedirs(macro_path)


TO_PLOT_FILES = [
             ("MMP12Cat_AHA_T1_ref_270510--MMP12Cat_Dive_T1_ref_peaks--RASmart(79.05).csv", "4GQL"),


             ("MMP12Cat_AHA_T1_ref_270510--MMP12Cat_NNGH_T1_ref_300707--RASmart(73.10).csv", "1Z3J"),

             ("CAIIZn_000_furo_03_170317--CAIIZn_100_furo_11_170317--RASmart(96.71).csv", "5EH8"),
             ("baz2a_phd_ref_renumbered--baz2a_phd_vs_10mer_8_1_renumbered--RASmart(83.02).csv", "5T8R"),

            ("Ube2T_ref_final--ube2t_em04_3mM_manual--RASmart(97.32).csv", "5NGZ"),
            ("baz2b_phd_ref_renumbered--baz2b_vs_5-mer_20_1_renumbered--RASmart(88.68).csv", "6FHQ")
            ]


for item in TO_PLOT_FILES:
    res_csv_file, pdb_file = item

    print("#########", res_csv_file,">>>" ,pdb_file)
    prot, with_lig, alg_ = res_csv_file.split("--")
    alg_ = alg_[:alg_.index(".csv")]

    ii = alg_.index('(')
    alg = alg_[:ii]

    print("\n", "///" * 20)
    print(res_csv_file)
    print(alg)
    csvdata = pd.read_csv(os.path.join(path, res_csv_file))
    csvdata = clean_data(csvdata)

    print(csvdata)

    high_csp_auto, low_csp_auto, u_auto, s_auto = get_high_low_CSPs(csvdata, "Automated", auto=True)
    print(high_csp_auto)
    print(low_csp_auto)


    high_csp_exp, low_csp_exp, u_exp, s_exp = get_high_low_CSPs(csvdata, "CSP (ppm)")

    file_path = os.path.join(macro_path, f"{prot}--{with_lig}[{alg_}].pml")


    pymol_script=f'''
fetch {pdb_file}
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, {pdb_file}

select highauto, (resi {high_csp_auto}) and pred
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi {low_csp_auto}) and pred
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi {high_csp_exp}) and {pdb_file}
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi {low_csp_exp}) and {pdb_file}
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect'''

    with open(file_path, 'w') as f:
        f.writelines(pymol_script)