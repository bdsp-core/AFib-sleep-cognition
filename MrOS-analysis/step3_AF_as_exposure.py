import datetime
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from myfunctions import *


def main(control_AHI=True, age_interaction=False):
    methods = ['adj_linreg',
            #'adj_bart',
            #'ipw',
            #'matching',
            #'msm',
            'dr',
        ]#TODO'dml', 'tmle'...
    random_state = 2023

    df = pd.read_excel('dataset.xlsx')
    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)

    A_col = 'A_AF_ECG'
    L_cols = sorted([x for x in df.columns if x.startswith('L_')])
    L_cols.remove('L_DPGDS15') # not interesting
    L_cols.remove('L_QLFXST51') # not interesting
    L_cols.remove('L_PASCORE') # not interesting
    L_cols.remove('L_EPEPWORT') # on the pathway
    L_cols.remove('L_MHCHF') # on the pathway
    L_cols.remove('L_MHMI') # on the pathway
    #L_caliper = {}
    print(L_cols)
    #['L_AHI3', 'L_AntiDep', 'L_Benzo', 'L_GIEDUC', 'L_HWBMI', 'L_MHBP', 'L_MHDIAB', 'L_MHSTRK', 'L_ODI3', 'L_RACE_AFR', 'L_RACE_ASI', 'L_RACE_HIS', 'L_RACE_OTH', 'L_TURSMOKE', 'L_VSAGE1', 'L_VSCPAP', 'L_hypoxemia_perc']

    Y_cols = [x for x in df.columns if x.startswith('M_')]# + [x for x in df.columns if x.startswith('Y_')]
    is_N2_N3 = lambda x: 'N2_N3' in x.upper() or 'N2N3' in x.upper() or 'N2+N3' in x.upper()
    Y_cols1 = [x for x in Y_cols if ((x.startswith('M_sp') or x.startswith('M_sw')) and is_N2_N3(x)) or 'coup' in x.lower()]
    Y_cols2 = [x for x in Y_cols if (not x.startswith('M_sp') and not x.startswith('M_sw')) and not is_N2_N3(x)]
    Y_cols = Y_cols1+Y_cols2
    to_remove_cols = [
        'M_EMG_env_level_N1', 'M_EMG_env_level_N2',
        'M_EMG_env_level_N3', 'M_EMG_env_level_R',
        'M_EMG_env_level_W',
        #'M_HB_apnea', 'M_HB_desat',
        'M_PLMI',
        'M_REM_density',
        'M_sw_slope_neg_slope_N2N3_C',
        'M_sw_slope_pos_slope_N2N3_C',
        'M_BAIPerc']
    to_remove_cols += [x for x in Y_cols if 'bp' in x and 'rel' in x]
    to_remove_cols += [x for x in Y_cols if 'bp' in x and 'slope' in x]
    Y_cols = sorted([x for x in Y_cols if x not in to_remove_cols]+['M_bp_delta_abs_slope_N3_C'])
    print(Y_cols)
    #['M_AI', 'M_macro_N1Perc', 'M_macro_N1Time', 'M_macro_N1toN1', 'M_macro_N1toN2', 'M_macro_N1toN3', 'M_macro_N1toR', 'M_macro_N1toW', 'M_macro_N2Perc', 'M_macro_N2Time', 'M_macro_N2toN1', 'M_macro_N2toN2', 'M_macro_N2toN3', 'M_macro_N2toR', 'M_macro_N2toW', 'M_macro_N3Perc', 'M_macro_N3Time', 'M_macro_N3toN1', 'M_macro_N3toN2', 'M_macro_N3toN3', 'M_macro_N3toR', 'M_macro_N3toW', 'M_macro_REMPerc', 'M_macro_REMTime', 'M_macro_RL', 'M_macro_RtoN1', 'M_macro_RtoN2', 'M_macro_RtoN3', 'M_macro_RtoR', 'M_macro_RtoW', 'M_macro_SFI', 'M_macro_SL', 'M_macro_TST', 'M_macro_WASO', 'M_macro_WtoN1', 'M_macro_WtoN2', 'M_macro_WtoN3', 'M_macro_WtoR', 'M_macro_WtoW',
    #'M_BAI', 'M_bp_alpha_abs_mean_N1_C', 'M_bp_alpha_abs_mean_N2_C', 'M_bp_alpha_abs_mean_N3_C', 'M_bp_alpha_abs_mean_R_C', 'M_bp_alpha_abs_mean_W_C', 'M_bp_delta_abs_mean_N1_C', 'M_bp_delta_abs_mean_N2_C', 'M_bp_delta_abs_mean_N3_C', 'M_bp_delta_abs_mean_R_C', 'M_bp_delta_abs_mean_W_C', 'M_bp_delta_abs_slope_N3_C', 'M_bp_theta_abs_mean_N1_C', 'M_bp_theta_abs_mean_N2_C', 'M_bp_theta_abs_mean_N3_C', 'M_bp_theta_abs_mean_R_C', 'M_bp_theta_abs_mean_W_C', 'M_sp_amp_N2N3_C', 'M_sp_dens_N2N3_C', 'M_sp_dur_N2N3_C', 'M_sp_dur_total_N2N3_C', 'M_sp_freq_N2N3_C', 'M_sp_sw_coupl_perc_C', 'M_sp_sym_N2N3_C', 'M_sw_amp_N2N3_C', 'M_sw_amp_neg_N2N3_C', 'M_sw_amp_pos_N2N3_C', 'M_sw_dur_perc_N2N3_C', 'M_sw_dur_total_N2N3_C', 'M_sw_freq_N2N3_C', 'M_sw_slope_neg_N2N3_C', 'M_sw_slope_pos_N2N3_C']
    #, 'M_HB_apnea', 'M_HB_desat'
    import pdb;pdb.set_trace()

    # get number of effective tests
    Y  = df[Y_cols].values
    Y2 = (Y-np.nanmean(Y,axis=0))/np.nanstd(Y,axis=0)
    knn = KNNImputer(n_neighbors=10)
    Y3 = knn.fit_transform(Y2)
    pca = PCA(n_components=0.95).fit(Y3)
    Neff = len(pca.explained_variance_ratio_)
    print(f'Neff = {Neff}')

    if control_AHI:
        suffix = ''
    else:
        suffix = '_notholdAHI'
        L_cols.remove('L_AHI3')
        Y_cols.append('L_AHI3')
    if age_interaction:
        suffix += '_age_interaction'
    df.loc[(df.EEGArtifactRatio>=0.05)|(df.NumMissingStage>0), 'M_BAI'] = np.nan
    df.loc[(df.EEGArtifactRatio>=0.05)|(df.NumMissingStage>0), 'M_BAIPerc'] = np.nan

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
            args = [Y.values,A.values,L.values]
            if method=='matching':
                args.append(L_caliper)
            if age_interaction:
                L_interaction_ids = L_cols.index('L_VSAGE1')
            else:
                L_interaction_ids = None
            effect, effect_ci, pval, Y0, Y1 = func(*args, L_interaction_ids=L_interaction_ids, random_state=random_state, verbose=True)

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
    results['sig_bonf'] = (results.pval<0.05/Neff).astype(int)
    for method in methods:
        ids = results.method==method
        results.loc[ids, 'sig_fdr_bh'] = multipletests(results.pval[ids].values, method='fdr_bh')[0].astype(int)
    results = results.sort_values('pval', ignore_index=True)
    print(results)

    results.to_excel(f'AF_as_exposure{suffix}2.xlsx', index=False)
    with open(f'AF_as_exposure_potential_outcomes{suffix}2.pickle', 'wb') as ff:
        pickle.dump(Y_As, ff)


if __name__=='__main__':
    #TODO argparse
    age_interaction = sys.argv[1].lower()=='age_interaction'
    main(age_interaction=age_interaction)

