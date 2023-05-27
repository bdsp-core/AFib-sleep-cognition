import numpy as np
import pandas as pd


def mediation_analysis(A, M, L, Y):
    R_path = r'D:\software\R-4.2.0\bin\Rscript'
    n_jobs = 8
    random_state = 2023

    model_Y_AML = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state)
    model_Y_AML.fit(np.c_[A,M,L], Y)


def main():
    df = pd.read_excel('dataset.xlsx')

    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)
    A_col = 'A_AF_ECG'
    L_cols = [
            'L_Age', 'L_ESS', 'L_educ', 'L_BMI',
            # 'L_BP', 'L_PASE',
            'L_QoL_SF12']
    M_cols = [x for x in df.columns if x.startswith('M_')]#TODO +AHI
    Y_cols = [x for x in df.columns if x.startswith('Y_')]

    results = []
    for yi, Y_col in enumerate(Y_cols):
        print(f'[{yi+1}/{len(Y_cols)}] {Y_col}')
        df_ = df[[A_col]+L_cols+[Y_col]].dropna()
        A = df_[A_col].values.astype(int)
        L = df_[L_cols].values.astype(float)
        Y = df_[Y_col].values.astype(float)
        te, te_ci = get_total_effect(A, L, Y)

        for mi, M_col in enumerate(M_cols):
            print(f'    [{mi+1}/{len(M_cols)}] {M_col}')
            df_ = df[[A_col]+L_cols+[M_col,Y_col]].dropna()
            N = len(df_)

            A = df_[A_col].values.astype(int)
            L = df_[L_cols].values.astype(float)
            M = df_[M_col].values.astype(float)
            Y = df_[Y_col].values.astype(float)

            effects = mediation_analysis(A, M, L, Y)

            effect, effect_bt = matching(A, L, Y, [L_clip[x] for x in L_cols], Nbt=Nbt, random_state=random_state, n_jobs=n_jobs)
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
    results.to_excel(f'AF_on_sleep_results_Nbt{Nbt}.xlsx', index=False)


if __name__=='__main__':
    main()

