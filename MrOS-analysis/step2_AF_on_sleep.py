import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from myfunctions import *


def main():
    methods = [#'adj_linreg',
            'adj_bart',
            #'ipw',
            'matching',
            'msm', 'dr']#TODO'dml', 'tmle'...
    random_state = 2023

    df = pd.read_excel('dataset.xlsx')

    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)
    A_col = 'A_AF_ECG'

    control_AHI = 'ahi4'
    if control_AHI=='ahi3':
        L_cols = ['L_AHI3']
        suffix = '_holdAHI3'
    elif control_AHI=='ahi4':
        L_cols = ['L_AHI4']
        suffix = '_holdAHI4'
    else:
        L_cols = []
        suffix = '_notholdAHI'
    L_cols.extend([
            'L_Age', 'L_ESS', 'L_educ', 'L_BMI',
            # 'L_BP', 'L_PASE',
            'L_QoL_SF12'])

    L_caliper = {
            'L_AHI3': 3,
            'L_AHI4': 3,
            'L_Age':5,
            'L_ESS':6,
            'L_educ':3,
            'L_BMI':3,
            #'L_BP':10,
            #'L_PASE':100,
            'L_QoL_SF12':20,
            }

    Y_cols = [x for x in df.columns if x.startswith('M_')]
    if control_AHI is None:
        Y_cols.append('L_AHI3')
        Y_cols.append('L_AHI4')
        Y_cols.append('L_PLMI')

    results = []
    for yi, Y_col in enumerate(Y_cols):
        print(f'[{yi+1}/{len(Y_cols)}] {Y_col}')
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
                'value(-)':Y0, 'value(+)':Y1,
                'effect':effect,
                'lb':effect_ci[0], 'ub':effect_ci[1],
                'pval':pval,
                }
            res_ = pd.DataFrame(data={k:[v] for k,v in res_.items()})
            print(res_)
            results.append(res_)

    results = pd.concat(results, axis=0, ignore_index=True)
    results['sig_bonf'] = (results.pval<0.05/len(results)).astype(int)
    results['sig_fdr_bh'] = multipletests(results.pval,method='fdr_bh')[0].astype(int)
    results = results.sort_values('pval', ignore_index=True)
    print(results)
    results.to_excel(f'AF_on_sleep{suffix}.xlsx', index=False)


if __name__=='__main__':
    main()

