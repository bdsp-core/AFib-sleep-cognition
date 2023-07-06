import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from myfunctions import *


def mediation_analysis(A, M, L, Y, M_A, Y_A, Y_M, method):
    n_jobs = 8
    random_state = 2023
    N = len(A)
    R_path = r'D:\software\R-4.2.0\bin\Rscript'

    # add augmented data
    for i in range(N):
        if A[i]==0:
            #01
            newA.append(0)
            newM.append(M_A[1][i])
            newY.append(Y_A[?])
            newL.append(L[i])
            #10
            #11
    model_Y_AML = MyBartRegressor(n_tree=100, R_path=R_path, verbose=False, n_jobs=n_jobs, random_state=random_state)
    model_Y_AML.fit(np.c_[A,M,L], Y)

    # Yab: E[Y(a,m(b))]
    Y00, Y00_bt = model_Y_AML.predict(np.c_[np.zeros(N),M_A[0],L], CI=True)
    Y01, Y01_bt = model_Y_AML.predict(np.c_[np.zeros(N),M_A[1],L], CI=True)
    Y10, Y10_bt = model_Y_AML.predict(np.c_[np.ones(N),M_A[0],L], CI=True)
    Y11, Y11_bt = model_Y_AML.predict(np.c_[np.ones(N),M_A[1],L], CI=True)
    model_Y_AML.clear()

    nde = Y10_bt - Y00_bt
    nie = Y11_bt - Y10_bt
    te  = Y11_bt - Y00_bt
    nde_perc = np.mean(nde/te, axis=1)*100
    nie_perc = np.mean(nie/te, axis=1)*100
    nde = nde.mean(axis=1)
    nie = nie.mean(axis=1)
    te = te.mean(axis=1)

    te_ci = np.percentile(te, (2.5,97.5))
    te_pval = 2*min((te<0).mean(), (te>0).mean())
    nde_ci = np.percentile(nde, (2.5,97.5))
    nde_pval = 2*min((nde<0).mean(), (nde>0).mean())
    nie_ci = np.percentile(nie, (2.5,97.5))
    nie_pval = 2*min((nie<0).mean(), (nie>0).mean())
    nde_perc_ci = np.percentile(nde_perc, (2.5,97.5))
    nie_perc_ci = np.percentile(nie_perc, (2.5,97.5))
    return (te.mean(), te_ci, te_pval,
           nde.mean(), nde_ci, nde_pval,
           nie.mean(), nie_ci, nie_pval,
           nde_perc.mean(), nde_perc_ci,
           nie_perc.mean(), nie_perc_ci,
           Y00, Y01, Y10, Y11)


def main():
    methods = [#'adj_linreg',
            'adj_bart',
            #'ipw',
            #'matching',
            #'msm',
            'dr']#TODO'dml', 'tmle'...
    random_state = 2023

    df = pd.read_excel('dataset.xlsx')

    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)
    A_col = 'A_AF_ECG'
    L_cols = [
            'L_Age', 'L_ESS', 'L_educ', 'L_BMI',
            # 'L_BP', 'L_PASE',
            'L_QoL_SF12']
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
    Y_cols = [x for x in df.columns if x.startswith('Y_')]
    #M_cols = [x for x in df.columns if x.startswith('M_')]+['L_AHI3','L_AHI4','L_PLMI']
    M_cols = [  # only the significant ones from AF-->sleep
        'M_sp_amp_N2_C',
        'M_sp_amp_N2N3_C',
        'M_macro_SFI',
        'M_macro_N1toW',
        'M_macro_N1toN1',
        'L_AHI4',
        'L_AHI3',
        ]

    results = []
    for yi, Y_col in enumerate(Y_cols):
        print(f'[{yi+1}/{len(Y_cols)}] {Y_col}')
        df_ = df[[A_col]+L_cols+[Y_col]].dropna()
        A = df_[A_col]
        L = df_[L_cols]
        Y = df_[Y_col]
        te = {}
        te_ci = {}
        te_pval = {}
        Y0 = {}
        Y1 = {}
        for m in methods:
            func = eval(m)
            np.random.seed(random_state)
            args = [A,L,Y]
            if m=='matching':
                args.append(L_caliper)
            te[m], te_ci[m], te_pval[m], Y0[m], Y1[m] = func(*args, random_state=random_state, verbose=False)
        print(f'total effect: {te}, p: {te_pval}')

        for mi, M_col in enumerate(M_cols):
            print(f'    [{mi+1}/{len(M_cols)}] {M_col}')
            df_ = df[[A_col]+L_cols+[M_col,Y_col]].dropna()
            N = len(df_)

            A = df_[A_col]
            L = df_[L_cols]
            M = df_[M_col]
            Y = df_[Y_col]

            for m in methods:
                func = eval(m)
                np.random.seed(random_state)
                args = [A,L,M]
                if m=='matching':
                    args.append(L_caliper)
                _, _, _, M_A0, M_A1 = func(*args, random_state=random_state, verbose=False)

                te2, te2_ci, te2_pval,\
                nde, nde_ci, nde_pval,\
                nie, nie_ci, nie_pval,\
                nde_perc, nde_perc_ci,\
                nie_perc, nie_perc_ci,\
                Y00, Y01, Y10, Y11 = mediation_analysis(A, M, L, Y, [M_A0,M_A1], m)

                res_ = {
                    'Yname':Y_col, 'Mname':M_col,
                    'method':m, 'N':len(A),
                    'N(-)':np.sum(A==0), 'N(+)':np.sum(A==1),
                    'value(--)':Y00.mean(), 'value(-+)':Y01.mean(),
                    'value(+-)':Y10.mean(), 'value(++)':Y11.mean(),
                    'te_Y_AL':te[m], 'te_Y_AL_lb':te_ci[m][0], 'te_Y_AL_ub':te_ci[m][1], 'te_Y_AL_pval':te_pval[m],
                    'te_sum':te2, 'te_sum_lb':te2_ci[0], 'te_sum_ub':te2_ci[1], 'te_sum_pval':te2_pval,
                    'nde':nde, 'nde_lb':nde_ci[0], 'nde_ub':nde_ci[1], 'nde_pval':nde_pval,
                    'nde_perc':nde_perc, 'nde_perc_lb':nde_perc_ci[0], 'nde_perc_ub':nde_perc_ci[1],
                    'nie':nie, 'nie_lb':nie_ci[0], 'nie_ub':nie_ci[1], 'nie_pval':nie_pval,
                    'nie_perc':nie_perc, 'nie_perc_lb':nie_perc_ci[0], 'nie_perc_ub':nie_perc_ci[1],
                    }
                res_ = pd.DataFrame(data={k:[v] for k,v in res_.items()})
                print(res_.iloc[0])
                results.append(res_)

    results = pd.concat(results, axis=0, ignore_index=True)
    #results['sig_bonf'] = (results.pval<0.05/len(results)).astype(int)
    #results['sig_fdr_bh'] = multipletests(results.pval,method='fdr_bh')[0].astype(int)
    #results = results.sort_values('pval', ignore_index=True)
    print(results)
    results.to_excel(f'mediation_result{suffix}.xlsx', index=False)


if __name__=='__main__':
    main()

