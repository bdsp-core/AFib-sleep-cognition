import datetime
import os
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm

R_path = r'D:\software\R-4.2.0\bin\Rscript'


def matching(A, L, Y, caliper, random_state=None, verbose=False):
    folder = os.getcwd()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    datapath = os.path.join(folder, f'data-{now}.csv')
    pd.concat([A,Y,L],axis=1).to_csv(datapath, index=False)
    resultpath = os.path.join(folder, f'result-{now}.csv')

    formula1 = A.name + '~' + '+'.join(L.columns)
    formula2 = Y.name + '~' + A.name+'*('+'+'.join(L.columns)+')'
    caliper_code = ','.join([f'{col}={caliper[col]}' for col in L.columns])

    if random_state is None:
        random_state = np.random.randint(10000)

    codepath = os.path.join(folder, f'code-{now}.R')
    with open(codepath, 'w') as ff:
        datapath2 = datapath.replace('\\','\\\\')
        resultpath2 = resultpath.replace('\\','\\\\')
        ff.write(
f"""library(MatchIt)
library(marginaleffects)

set.seed({random_state})
df <- read.csv("{datapath2}")

res <- matchit(
    {formula1}, data=df, method="full",
    distance="glm", estimand="ATE",
    caliper=c({caliper_code}),
    std.caliper=FALSE,
    verbose=FALSE
)
m.data <- match.data(res)

fit <- lm({formula2}, data=m.data, weights=weights)
res <- comparisons(fit, variables="{A.name}",
               vcov=~subclass, newdata=m.data, wts="weights")
res.avg <- avg_comparisons(fit, variables="{A.name}",
               vcov=~subclass, newdata=m.data, wts="weights")
res.avg <- data.frame(res.avg)
res.avg$high <- weighted.mean(res$predicted_hi, res$weights)
res.avg$low  <- weighted.mean(res$predicted_lo, res$weights)
write.csv(res.avg, "{resultpath2}", row.names=FALSE)
""")
    subprocess.check_call([R_path, codepath])#,
    #    stdout=subprocess.STDOUT if verbose else subprocess.DEVNULL, stderr=subprocess.STDOUT)
    res = pd.read_csv(resultpath)
    
    os.remove(codepath)
    os.remove(datapath)
    os.remove(resultpath)

    return res.loc[0,'estimate'], (res.loc[0,'conf.low'], res.loc[0,'conf.high']), res.loc[0,'p.value'], res.loc[0,'low'], res.loc[0,'high']  # not per-sample prediction


class MyBartRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_tree=100, R_path='Rscript', verbose=False, n_jobs=1, random_state=None):
        self.n_tree = n_tree
        self.R_path = R_path
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def clear(self):
        if hasattr(self, 'modelpath') and os.path.exists(self.modelpath):
            os.remove(self.modelpath)

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
        codepath = os.path.join(folder, f'code-{now}.R')
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
        subprocess.check_call([self.R_path, codepath])#,
        #    stdout=subprocess.STDOUT if self.verbose else subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
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
        subprocess.check_call([self.R_path, codepath])#,
        #    stdout=subprocess.STDOUT if self.verbose else subprocess.DEVNULL, stderr=subprocess.STDOUT)
        yp = pd.read_csv(resultpath).values

        os.remove(resultpath)
        os.remove(codepath)
        os.remove(datapath)

        ypmean = yp.mean(axis=0)
        if CI is None:
            return ypmean
        elif type(CI)==bool and CI:
            return ypmean, yp
        else:
            lb = (100-CI)/2
            ub = lb+CI
            return ypmean, np.percentile(yp, (lb,ub), axis=0)


def adj_bart(A, L, Y, random_state=None, verbose=False):
    n_jobs = 8
    N = len(A)

    model_Y_AL = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
    model_Y_AL.fit(np.c_[A,L], Y)

    Y_A01, Y_A01_bt = model_Y_AL.predict(np.r_[
        np.c_[np.zeros(N),L],
        np.c_[np.ones(N),L],
        ], CI=True)
    Y_A0 = Y_A01[:N]; Y_A1 = Y_A01[N:]
    Y_A0_bt = Y_A01_bt[:,:N]; Y_A1_bt = Y_A01_bt[:,N:]

    Y_A1_bt = Y_A1_bt.mean(axis=1)
    Y_A0_bt = Y_A0_bt.mean(axis=1)
    te = Y_A1.mean() - Y_A0.mean()
    te_bt = Y_A1_bt - Y_A0_bt
    ci = np.percentile(te_bt, (2.5,97.5), axis=0)
    pval = 2*min((te_bt<0).mean(), (te_bt>0).mean())
    model_Y_AL.clear()
    return te, ci, pval, Y_A0, Y_A1


def adj_linreg(A, L, Y, random_state=None, verbose=False):
    N = len(A)
    model_Y_AL = sm.OLS(Y,np.c_[A,L,np.ones(N)])
    res = model_Y_AL.fit()
    te =  res.params[0]
    ci =  res.conf_int().values[0]
    pval =  res.pvalues[0]
    Y_A0 = res.predict(np.c_[np.zeros(N),L,np.ones(N)])
    Y_A1 = res.predict(np.c_[np.ones(N),L,np.ones(N)])
    return te, ci, pval, Y_A0, Y_A1


def ipw(A, L, Y, random_state=None, verbose=False):
    Nbt = 10000
    n_jobs = 8
    np.random.seed(random_state)
    N = len(A)
    A = A.values; L = L.values; Y = Y.values
    def _func(bti):
        if bti==0:
            ids = np.arange(N)
        else:
            ids = np.random.choice(N,N,replace=True)
        Abt = A[ids]
        Lbt = L[ids]
        Ybt = Y[ids]
        pmodel = LogisticRegression(penalty=None, max_iter=1000).fit(Lbt,Abt)
        Ap = pmodel.predict_proba(Lbt)[:,1]
        Y_A1 = Ybt*(Abt==1).astype(float)/Ap
        Y_A0 = Ybt*(Abt==0).astype(float)/(1-Ap)
        return Y_A1.mean() - Y_A0.mean(), Y_A0, Y_A1
    with Parallel(n_jobs=n_jobs, verbose=False) as par:
        res = par(delayed(_func)(bti) for bti in tqdm(range(Nbt+1), disable=not verbose))
    te, Y_A0, Y_A1 = res[0]
    te_bt = np.array([x[0] for x in res[1:]])
    ci = np.percentile(te_bt, (2.5,97.5))
    pval = 2*min((te_bt<0).mean(), (te_bt>0).mean())
    return te, ci, pval, Y_A0, Y_A1


def msm(A, L, Y, random_state=None, verbose=False):
    Nbt = 10000
    n_jobs = 8
    np.random.seed(random_state)
    N = len(A)
    A = A.values; L = L.values; Y = Y.values
    def _func(bti):
        if bti==0:
            ids = np.arange(N)
        else:
            ids = np.random.choice(N,N,replace=True)
        Abt = A[ids]
        Lbt = L[ids]
        Ybt = Y[ids]

        pmodel = LogisticRegression(penalty=None, max_iter=1000).fit(Lbt,Abt)
        Apred = pmodel.predict_proba(Lbt)
        Aprop = np.zeros((N,2)); Aprop[:,0] = (Abt==0).mean(); Aprop[:,1] = (Abt==1).mean()
        sw = Aprop[range(N),Abt.astype(int)] / Apred[range(N),Abt.astype(int)]

        model = LinearRegression().fit(Abt.reshape(-1,1), Ybt, sample_weight=sw)
        Y_A0 = model.predict(np.array([[0]]))[0]
        Y_A1 = model.predict(np.array([[1]]))[0]
        return Y_A1-Y_A0, Y_A0, Y_A1
    with Parallel(n_jobs=n_jobs, verbose=False) as par:
        res = par(delayed(_func)(bti) for bti in tqdm(range(Nbt+1), disable=not verbose))
    te, Y_A0, Y_A1 = res[0]
    te_bt = np.array([x[0] for x in res[1:]])
    ci = np.percentile(te_bt, (2.5,97.5))
    pval = 2*min((te_bt<0).mean(), (te_bt>0).mean())
    return te, ci, pval, Y_A0, Y_A1  # not per-sample prediction


def dr(A, L, Y, random_state=None, verbose=False):
    Nbt = 10000
    n_jobs = 8
    np.random.seed(random_state)
    N = len(A)
    A = A.values; L = L.values; Y = Y.values
    def _func(bti):
        if bti==0:
            ids = np.arange(N)
        else:
            ids = np.random.choice(N,N,replace=True)
        Abt = A[ids]
        Lbt = L[ids]
        Ybt = Y[ids]

        pmodel = LogisticRegression(penalty=None, max_iter=1000).fit(Lbt,Abt)
        ymodel = LinearRegression().fit(np.c_[Abt,Lbt], Ybt)
        #ymodel = MyBartRegressor(n_tree=100, R_path=R_path, n_jobs=n_jobs, random_state=random_state, verbose=verbose).fit(np.c_[Abt,Lbt],Ybt)

        Ap = pmodel.predict_proba(Lbt)
        yp0 = ymodel.predict(np.c_[np.zeros(N),Lbt])
        yp1 = ymodel.predict(np.c_[np.ones(N),Lbt])
        Y_A0 = (Ybt-yp0)*(Abt==0).astype(float)/Ap[:,0] + yp0
        Y_A1 = (Ybt-yp1)*(Abt==1).astype(float)/Ap[:,1] + yp1
        return Y_A1.mean()-Y_A0.mean(), Y_A0, Y_A1
    with Parallel(n_jobs=n_jobs, verbose=False) as par:
        res = par(delayed(_func)(bti) for bti in tqdm(range(Nbt+1), disable=not verbose))
    te, Y_A0, Y_A1 = res[0]
    te_bt = np.array([x[0] for x in res[1:]])
    ci = np.percentile(te_bt, (2.5,97.5))
    pval = 2*min((te_bt<0).mean(), (te_bt>0).mean())
    return te, ci, pval, Y_A0, Y_A1

