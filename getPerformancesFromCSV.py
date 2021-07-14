import os
import pandas as pd
import pathlib
import sys

_path = pathlib.Path('.')

all_files = os.listdir(_path)
print(_path)
print(str(_path.parent))
print(all_files)

experiment_files = []
for file in all_files:
    if file.endswith(".csv"):
        experiment_files.append(file)


algorithm_names = ['SDSmart', 'SD', 'RA', 'RASmart']
results_df = pd.DataFrame(columns=['free', 'with_ligand']+algorithm_names)


def get_alg_from_name(name):
        print(name)
        if name.find('SDSmart') >= 0:
            return 'SDSmart'

        elif name.find('RASmart') >= 0:
            return 'RASmart'

        elif name.find('RA') >= 0:
            return 'RA'

        elif name.find('SD') >= 0:
            return 'SD'

        else:
            Exception



for file in experiment_files:
    if len(file.split('--')) == 3:
        free, with_lig, alg_res = file.split('--')

        df = pd.read_csv(file, sep=";",  index_col=0)
        Peaks = df.index.tolist()
        AssPeaks = df['assigned_to'].tolist()
        assert len(Peaks) == len(AssPeaks)

        wrong = 0.
        ok = 0.

        for P_key in Peaks:
            assigned_to = df.loc[P_key, 'assigned_to']
            if assigned_to != 'na':
                #print(P_key, assigned_to)
                if str(P_key) == str(assigned_to):
                    ok += 1
                else:
                    wrong+=1

        print("ok: ", ok, " wrong: ", wrong)

        '''
        for P_idx, P_key in enumerate(Peaks):
            if P_key != 'na':
                #if Peaks[P_i] == AssPeaks[P_i]:
                #    ok+=1
                #else:
                #    wrong+=1
                #else: # P_key risulta non assegnato
                if P_key not in AssPeaks:
                    # e infatto non c'è tra i newPeaks
                    ok+=1
                else:
                    # ma invece è stato assegnato
                    wrong+=1
        '''

        acc = ok /(ok + wrong)
        idx = free + "--"+with_lig
        results_df.loc[idx, 'free'] = free
        results_df.loc[idx, 'with_ligand'] = with_lig
        alg_ = get_alg_from_name(alg_res)
        results_df.loc[idx, alg_] = acc


        print(idx)
        print("=>", file, "[", acc, "]", alg_)

print(results_df)
results_df.to_csv("results_110221.csv")

