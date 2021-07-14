import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
print(os.getcwd())

path = "/Users/vincenzo/CERM/PeaksIdentification/testDundee110221_conversed"

files = os.listdir(path)
print(files)

plot_path = os.path.join(path, "CSP_plots20210610")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

for file in files:
    if file.endswith(".csv"):
        prot, with_lig, alg = file.split("--")

        ii = alg.index('(')
        alg = alg[:ii]

        if alg=="RASmart":
            print("///" * 20)
            print(file)
            print(alg)

            csvdata = pd.read_csv(os.path.join(path, file))

            for col in csvdata.columns:
                print(">>>", col)
                before = csvdata[col]
                csvdata[col] = csvdata[col].astype(str).replace(['--', "--"], None)
                after = csvdata[col]

                if '--' in csvdata[col].values:
                    print(">>> YEEESSSSS")
                    #print(before)
                    #print(after)
                    #print(after[after=='--'])
                    print("mmm...")
                    after_list = after.to_list()
                    for j, item in enumerate(after_list):
                        #print(item, item.encode(), '--'.encode())
                        if item.encode() == '--'.encode():
                            #print("...beccato...")
                            after_list[j] = None
                    #print(after_list)
                    csvdata[col] = after_list

            x1 = csvdata['Residue Number'].astype(int)
            y1 = csvdata['CSP (ppm)'].astype(float)

            #x2 = csvdata['Residue Number'].astype(int)
            y2 = csvdata['Automated'].astype(float)


            plt.plot(x1, y2, 's', c='black', markersize=5, markeredgewidth=0.5) # square --> automated
            #plt.vlines(x1, [0], y2, colors='black', linewidth=0.5)
            plt.vlines(x1, [0], y2, colors='green', linewidth=1, alpha=1)
            U = y2.mean()
            S = y2.std()

            gt4sigma_idxs = (y2 > U + 4 * S).to_numpy().nonzero()[0]
            lt4sigma_idxs = [idx for idx in csvdata.index.values if idx not in gt4sigma_idxs]
            csvdata_senza = csvdata.iloc[lt4sigma_idxs]
            mu_exp = y1.mean()
            std_exp = y1.std()
            mu2 = csvdata_senza['Automated'].astype(float).mean()
            std2 = csvdata_senza['Automated'].astype(float).std()

            #plt.axhline(y=U + 4* S, color='r', linestyle='--')
            plt.axhline(y=mu2 + std2,  color='green', linestyle='--')
            #plt.axhline(y=mu, color='r', linestyle='-')
            plt.axhline(y=mu_exp + std_exp, color='orange', linestyle='--')


            plt.plot(x1, y1, 'o', c='white', markersize=4, markeredgecolor="black", markeredgewidth=0.7) # circle --> experimental
            plt.vlines(x1, [0], y1, colors='black', linewidth=0.5)

            #plt.title(f"{prot}, {with_lig} - [{alg}]", fontsize=10)

            plt.xlabel("Residue Number")
            plt.ylabel("CSP (ppm)")


            # PLOT MEAN VALUES

            # MARK CPS WITHIN 2 SIGMAS


            file = file.split('.csv')[0]
            fig_file = os.path.join(plot_path, file+".png")

            if os.path.exists(fig_file):
                print("Exists!")
                os.remove(fig_file)
            else:
                print("NO")
            plt.tight_layout()
            plt.savefig(fig_file, dpi=199)
            plt.show()
            #sys.exit()


