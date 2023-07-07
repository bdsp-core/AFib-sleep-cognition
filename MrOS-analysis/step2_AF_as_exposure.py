import datetime
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from myfunctions import *


def main(control_AHI):
    methods = ['adj_linreg',
            'adj_bart',
            #'ipw',
            #'matching',
            #'msm',
            'dr'
    ]#TODO'dml', 'tmle'...
    random_state = 2023

    df = pd.read_excel('dataset.xlsx')
    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)

    A_col = 'A_AF_ECG'
    L_cols = [x for x in df.columns if x.startswith('L_')]
    print(L_cols)
    #L_caliper = {}
    Y_cols = [x for x in df.columns if x.startswith('M_')] + [x for x in df.columns if x.startswith('Y_')]

    if control_AHI.lower()=='control_ahi':
        L_cols.append('M_AHI3')
        Y_cols.remove('M_AHI3')
        suffix = '_holdAHI'
    else:
        suffix = '_notholdAHI'

    results = []
    Y_As = {}
    for yi, Y_col in enumerate(Y_cols):
        now = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
        print(f'[{yi+1}/{len(Y_cols)}] {Y_col} {now}')
        df_ = df[[A_col]+L_cols+[Y_col]].dropna()
        N = len(df_)

        A = df_[A_col]
        L = df_[L_cols]
        Y = df_[Y_col]

        for mdi, method in enumerate(methods):
            func = eval(method)
            np.random.seed(random_state)
            args = [A,L,Y]
            if method=='matching':
                args.append(L_caliper)
            effect, effect_ci, pval, Y0, Y1 = func(*args, random_state=random_state, verbose=True)

            res_ = {
                'Name':Y_col,'method':method,
                'N':len(A), 'N(-)':np.sum(A==0), 'N(+)':np.sum(A==1),
                'value(-)':Y0.mean(), 'value(+)':Y1.mean(),
                'effect':effect,
                'lb':effect_ci[0], 'ub':effect_ci[1],
                'pval':pval,
                }
            res_ = pd.DataFrame(data={k:[v] for k,v in res_.items()})
            print(res_)
            results.append(res_)
            Y_As[(Y_col,method)] = [Y0,Y1]

    results = pd.concat(results, axis=0, ignore_index=True)
    results['sig_bonf'] = (results.pval<0.05/len(Y_cols)).astype(int)
    for method in methods:
        ids = results.method==method
        results.loc[ids, 'sig_fdr_bh'] = multipletests(results.pval[ids].values, method='fdr_bh')[0].astype(int)
    results = results.sort_values('pval', ignore_index=True)
    print(results)

    results.to_excel(f'AF_as_exposure{suffix}.xlsx', index=False)
    with open(f'AF_as_exposure_potential_outcomes{suffix}.pickle', 'wb') as ff:
        pickle.dump(Y_As, ff)


if __name__=='__main__':
    control_AHI = sys.argv[1]
    main(control_AHI)

