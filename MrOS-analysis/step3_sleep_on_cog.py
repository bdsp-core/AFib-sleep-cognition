import re
from itertools import product
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from myfunctions import MyBartRegressor


def adj_bart2(A, L, Y, random_state=None, verbose=False):
    """
    continuous A
    the first dimension of L is binary, therefore fit
    E[Y|A,L_1=0,L_2:D] and E[Y|A,L_1=1,L_2:D] separately
    """
    N = len(A)
    A = A.values
    L = L.values
    Y = Y.values

    R_path = 'Rscript'#r'D:\software\R-4.2.0\bin\Rscript'
    n_jobs = 16

    ids0 = np.where(L[:,0]==0)[0]
    model_Y_AL0 = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
    model_Y_AL0.fit(np.c_[A,L[:,1:]][ids0], Y[ids0])
    ids1 = np.where(L[:,0]==1)[0]
    model_Y_AL1 = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
    model_Y_AL1.fit(np.c_[A,L[:,1:]][ids1], Y[ids1])

    nA = 20
    As = np.linspace(A.min(), A.max(), nA)

    Y_A = np.zeros((nA,N)); Y_A_bt = np.zeros((1000,nA,N))

    y, ybt = model_Y_AL0.predict(
            np.concatenate([np.c_[np.zeros(len(ids0))+a,L[:,1:][ids0]] for a in As], axis=0),
            CI=True)
    Y_A[:,ids0] = y.reshape(nA,-1)
    Y_A_bt[:,:,ids0] = ybt.reshape(1000,nA,-1)

    y, ybt = model_Y_AL1.predict(
            np.concatenate([np.c_[np.zeros(len(ids1))+a,L[:,1:][ids1]] for a in As], axis=0),
            CI=True)
    Y_A[:,ids1] = y.reshape(nA,-1)
    Y_A_bt[:,:,ids1] = ybt.reshape(1000,nA,-1)

    model_Y_AL0.clear()
    model_Y_AL1.clear()
    return As, Y_A, np.percentile(Y_A_bt, (2.5,97.5), axis=0)


def main():
    method = 'adj_bart'
    random_state = 2023
    suffix = ''

    df = pd.read_excel('dataset.xlsx')

    A_cols = [x for x in df.columns if x.startswith('M_')]
    A_cols.append('L_AHI3')
    A_cols.append('L_AHI4')
    A_cols.append('L_PLMI')

    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)
    L_cols = ['A_AF_ECG',
            'L_Age', 'L_ESS', 'L_educ', 'L_BMI',
            # 'L_BP', 'L_PASE',
            'L_QoL_SF12']

    Y_cols = [x for x in df.columns if x.startswith('Y_')]

    results = []
    all_Y_As = {}
    cc = 0
    Niter = len(A_cols)*len(Y_cols)
    for A_col, Y_col in product(A_cols, Y_cols):
        #print(f'[{yi+1}/{len(Y_cols)}] {Y_col}')
        cc += 1
        print(f'[{cc}/{Niter}: {A_col}, {Y_col}')
        df_ = df[[A_col]+L_cols+[Y_col]].dropna()
        N = len(df_)

        A = df_[A_col]
        L = df_[L_cols]
        Y = df_[Y_col]

        As, Y_As, Y_As_ci = adj_bart2(A,L,Y, random_state=random_state, verbose=True)

        res_ = {
            'Aname':A_col, 'Yname':Y_col, 'method':method,
            'N':len(A), 
            }
        res_ = pd.DataFrame(data={k:[v] for k,v in res_.items()})
        for ai, a in enumerate(As):
            res_[f'A{ai}'] = a
        for ai, a in enumerate(As):
            res_[f'Y{ai}'] = Y_As[ai].mean()
        for ai, a in enumerate(As):
            res_[f'Y_lb{ai}'] = Y_As_ci[0][ai].mean()
        for ai, a in enumerate(As):
            res_[f'Y_ub{ai}'] = Y_As_ci[1][ai].mean()
        print(res_)
        results.append(res_)
        all_Y_As[(A_col,Y_col,method,'As')] = As
        all_Y_As[(A_col,Y_col,method,'Y_As')] = Y_As
        all_Y_As[(A_col,Y_col,method,'Y_As_ci')] = Y_As_ci

        if cc%5==0 or cc==Niter:
            all_results = pd.concat(results, axis=0, ignore_index=True)
            #print(all_results)
            all_results.to_excel(f'sleep_on_cog{suffix}.xlsx', index=False)
            with open(f'sleep_on_cog{suffix}.pickle', 'wb') as ff:
                pickle.dump(all_Y_As, ff)

    # get slope pvalue
    df = pd.concat(results, axis=0, ignore_index=True)
    for i in tqdm(range(len(df))):
        As = df[[x for x in df.columns if re.match(r'A\d+',x)]].iloc[i].values
        Ys = df[[x for x in df.columns if re.match(r'Y\d+',x)]].iloc[i].values
        rho, pval = spearmanr(As, Ys)
        df.loc[i, 'rho'] = rho
        df.loc[i, 'pval'] = pval
    df = df.sort_values('pval', ignore_index=True)
    #TODO bonferoni for each method separately
    df['SigBonf'] = (df.pval<0.05/len(df)).astype(int)
    cols=['Aname', 'Yname','rho', 'pval','SigBonf', 'method', 'N']
    cols = cols + [x for x in df.columns if x not in cols]
    df = df[cols]
    print(df)
    df.to_excel(f'sleep_on_cog{suffix}.xlsx', index=False)


if __name__=='__main__':
    main()

