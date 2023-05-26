import datetime
from itertools import product
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin


class MyBartRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_tree=100, R_path='Rscript', verbose=False, n_jobs=1, random_state=None):
        self.n_tree = n_tree
        self.R_path = R_path
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        folder = os.getcwd()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        datapath = os.path.join(folder, f'data-{now}.csv')
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        pd.DataFrame(data=np.c_[X,y,sample_weight]).to_csv(datapath, index=False)

        if self.random_state is None:
            self.random_state = np.random.randint(10000)

        self.modelpath = os.path.join(folder, f'model-{now}.rds')
        codepath = folder+f'code-{now}.R'
        with open(codepath, 'w') as ff:
            datapath2 = datapath.replace('\\','\\\\')
            modelpath2 = self.modelpath.replace('\\','\\\\')
            ff.write(
f"""library(BART)
set.seed({self.random_state})
df <- read.csv("{datapath2}")
X <- df[,1:(ncol(df)-2)]
y <- df[,ncol(df)-1]
w <- df[,ncol(df)]
model <- wbart(X, y, w=w, ntree={self.n_tree})
saveRDS(model, file="{modelpath2}")
""")
        subprocess.check_call([self.R_path, codepath],
            stdout=subprocess.STDOUT if self.verbose else subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        os.remove(codepath)
        os.remove(datapath)
        
        return self

    def predict(self, X, CI=None):
        folder = os.getcwd()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        datapath = os.path.join(folder, f'data-{now}.csv')
        pd.DataFrame(data=X).to_csv(datapath, index=False)

        resultpath = os.path.join(folder, f'result-{now}.csv')
        codepath = os.path.join(folder, f'code-{now}.R')
        with open(codepath, 'w') as ff:
            datapath2 = datapath.replace('\\','\\\\')
            modelpath2 = self.modelpath.replace('\\','\\\\')
            resultpath2 = resultpath.replace('\\','\\\\')
            ff.write(
f"""library(BART)
set.seed({self.random_state})
df <- read.csv("{datapath2}")
model <- readRDS(file="{modelpath2}")
yp <- predict(model, df, mc.cores={self.n_jobs})
write.csv(yp, file="{resultpath2}", row.names=FALSE)
""")
        subprocess.check_call([self.R_path, codepath],
            stdout=subprocess.STDOUT if self.verbose else subprocess.DEVNULL, stderr=subprocess.STDOUT)
        yp = pd.read_csv(resultpath).values

        os.remove(resultpath)
        os.remove(codepath)
        os.remove(datapath)

        if CI is None:
            return yp.mean(axis=0)
        else:
            lb = (100-CI)/2
            ub = lb+CI
            return yp.mean(axis=0), np.percentile(yp, (lb,ub), axis=0)


def mediation_analysis(A, M, L, Y):
    R_path = r'D:\software\R-4.2.0\bin\Rscript'
    n_jobs = 8
    random_state = 2023

    model_Y_AML = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state)
    model_Y_AML.fit(np.c_[A,M,L], Y)

    model_Y_AL = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state)
    model_Y_AL.fit(np.c_[A,L], Y)
    import pdb;pdb.set_trace()

    total


def main():

    df = pd.read_excel('dataset.xlsx')

    #df['A_AF_ECG'] = ((df.A_AF_history.fillna(0).values+df.A_AF_ECG.values)>0).astype(int)
    A_col = 'A_AF_ECG'
    L_cols = [
            'L_Age', 'L_ESS', 'L_educ', 'L_BMI',
            # 'L_BP', 'L_PASE',
            'L_QoL_SF12']
    M_cols = [x for x in df.columns if x.startswith('M_')]
    Y_cols = [x for x in df.columns if x.startswith('Y_')]

    results = []
    for yi, (M_col, Y_col) in enumerate(product(M_cols, Y_cols)):
        print(f'[{yi+1}/{len(M_cols)*len(Y_cols)}] {M_col}, {Y_col}')
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

