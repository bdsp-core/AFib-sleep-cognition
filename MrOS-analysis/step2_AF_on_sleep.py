from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from statsmodels.stats.weightstats import ttest_ind


def matching(A, L, Y, L_clip, K):
    trt_class = 1
    ctl_class = 0
    trt_ids = np.where(A==trt_class)[0]
    ctl_ids = np.where(A==ctl_class)[0]

    L2 = (L-L.mean(axis=0))/L.std(axis=0)
    pmodel = LogisticRegression(penalty='none').fit(L2,A)
    pscore = pmodel.predict_proba(L2)[:,1]

    matched_ctl_ids = []
    weights = []
    Nmatches = []
    for idx in trt_ids:
        ids = ctl_ids
        for j in range(L.shape[1]):
            ids = ids[ (L[ids,j] >= L[idx,j]+L_clip[j][0])&(L[ids,j] <= L[idx,j]+L_clip[j][1]) ]
        #dists = np.sum((L2[idx]-L2[ids])**2, axis=1)
        dists = np.abs(pscore[idx]-pscore[ids])
        ids = ids[np.argsort(dists)[:K]]
        Nmatch = len(ids)
        Nmatches.append(Nmatch)
        if Nmatch == 0:
            #dists = np.sum((L2[idx]-L2[ctl_ids])**2, axis=1)
            dists = np.abs(pscore[idx]-pscore[ctl_ids])
            ids = ctl_ids[np.argsort(dists)[:1]]
            Nmatch = 1
        matched_ctl_ids.extend(ids)
        weights.extend([1/Nmatch]*Nmatch)

    #print(np.mean(np.array(Nmatches)==0))
    ids = np.r_[trt_ids, matched_ctl_ids]
    x = np.c_[A,L][ids]
    y = Y[ids]
    w = np.r_[np.ones_like(trt_ids), np.array(weights)]

    ids0 = x[:,0]==0
    ids1 = x[:,0]==1
    #pvals = []
    for j in range(1,x.shape[1]):
        pval = ttest_ind(x[:,j][ids0], x[:,j][ids1], usevar='unequal', weights=(w[ids0],w[ids1]))[1]
        #assert pval>0.05, (j,pval)#pvals.append(pval)
        if pval<0.05:
            return None

    model = LinearRegression().fit(x, y, w)
    #model = BayesianRidge().fit(x, y, w)
    return model.coef_[0]


def main():
    Nbt = 10000
    random_state = 2023
    n_jobs = 8
    K = 5

    df = pd.read_excel('dataset.xlsx')

    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)
    A_col = 'A_AF_ECG'

    control_AHI = None
    if control_AHI=='ahi3':
        L_cols = ['L_AHI3%']
        suffix = '_holdAHI3'
    elif control_AHI=='ahi4':
        L_cols = ['L_AHI4%']
        suffix = '_holdAHI4'
    else:
        L_cols = []
        suffix = '_notholdAHI'
    L_cols.extend([
            'L_Age', 'L_ESS', 'L_educ', 'L_BMI',
            # 'L_BP', 'L_PASE',
            'L_QoL_SF12'])

    Y_cols = [x for x in df.columns if x.startswith('M_')]
    if control_AHI is None:
        Y_cols.append('L_AHI3%')
        Y_cols.append('L_AHI4%')
        Y_cols.append('L_PLMI')

    L_clip = {
            'L_AHI3%': (-3,3),
            'L_AHI4%': (-3,3),
            'L_Age':(-5,5),
            'L_ESS':(-6,6),
            'L_educ':(-3,3),
            'L_BMI':(-3,3),
            #'L_BP':(-10,10),
            #'L_PASE':(-100,100),
            'L_QoL_SF12':(-20,20),
            }
    L_clip = [L_clip[x] for x in L_cols]

    results = []
    for yi, Y_col in enumerate(Y_cols):
        print(f'[{yi+1}/{len(Y_cols)}] {Y_col}')
        df_ = df[[A_col]+L_cols+[Y_col]].dropna()
        N = len(df_)

        A = df_[A_col].values.astype(int)
        L = df_[L_cols].values.astype(float)
        Y = df_[Y_col].values.astype(float)

        np.random.seed(random_state)
        def _fit(bti):
            ids = np.arange(N) if bti==0 else np.random.choice(N,N,replace=True)
            return matching(A[ids], L[ids], Y[ids], L_clip, K)
        with Parallel(n_jobs=n_jobs, verbose=False) as par:
            res_bt = par(delayed(_fit)(bti) for bti in tqdm(range(Nbt+1), disable=False))
        import pdb;pdb.set_trace()
        effect = res_bt[0]
        effect_bt = np.array([x for x in res_bt[1:] if x is not None])
        effect_lb, effect_ub = np.percentile(effect_bt, (2.5, 97.5))
        pval = 2*min((effect_bt<0).mean(), (effect_bt>0).mean())
        
        value_pos = Y[A==1].mean()
        res_ = {
            'Name':Y_col,
            'N':len(A),
            'N(-)':np.sum(A==0),
            'N(+)':np.sum(A==1),
            'value(-)':value_pos-effect,
            'value(+)':value_pos,
            'effect':effect,
            'lb':effect_lb,
            'ub':effect_ub,
            'pval':pval,
            }
        res_ = pd.DataFrame(data={k:[v] for k,v in res_.items()})
        print(res_)
        results.append(res_)

    results = pd.concat(results, axis=0, ignore_index=True)
    results['sig'] = (results.pval<0.05/len(results)).astype(int)
    results = results.sort_values('pval', ignore_index=True)
    print(results)
    results.to_excel(f'AF_on_sleep_results_Nbt{Nbt}{suffix}.xlsx', index=False)


if __name__=='__main__':
    main()

