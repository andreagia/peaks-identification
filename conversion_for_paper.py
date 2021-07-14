
import numpy as np
import pandas as pd

import os
import argparse
import sys


parser = argparse.ArgumentParser(description="Automatic peaks assignement tool")

parser.add_argument(
    "--path",
    default='.')


def clean(s):
    if str(s).isnumeric() or str(s)=='--':
        #print(s, " --> NUMERIC or '--' ")
        return s
    else:
        #print(s, " --> STRING ",s[1:])
        return s[1:]


def main(args):
    cwd = os.path.join(os.getcwd(), args.path)
    new_dir = cwd+"_conversed"

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    files = os.listdir(args.path)

    for file in files:
        if file.endswith(".csv"):

            if len(file.split("--")) == 3:

                file_csv_path = os.path.join(cwd, file)
                print(file_csv_path)
                df = pd.read_csv(file_csv_path, sep=";")

                #print(df)
                df = df[['order', 'real_shift', 'assigned_to', 'est_shift']].replace([-1., 'na', np.nan],'--')

                #df['assigned_to'].apply(lambda x: x[1:] if x[0].isdigit() else x)
                df['assigned_to'] = df['assigned_to'].apply(clean)
                #print(df['assigned_to'])

                df = df.rename(columns={'order': 'Residue Number',
                                        'real_shift': 'CSP (ppm)',
                                        'assigned_to': 'Pred_assign',
                                        'est_shift': 'Automated'})

                #print(df)

                df.to_csv(new_dir+"/"+file, index=False)





if __name__ == "__main__":
        args = parser.parse_args()
        main(args)
