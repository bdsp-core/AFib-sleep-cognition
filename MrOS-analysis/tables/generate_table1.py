import pandas as pd

df = pd.read_excel('../dataset.xlsx')

ids0=df.A_AF_ECG==0
ids1=df.A_AF_ECG==1

print(f'n(AFib+) = {ids1.sum()}\tn(AFib-) = {ids0.sum()}')

col = 'L_VSAGE1'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].min()}-{df[col][ids1].max()})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].min()}-{df[col][ids0].max()})')

print('Race')
df['L_RACE_White'] = ((df.L_RACE_AFR==0)&(df.L_RACE_ASI==0)&(df.L_RACE_HIS==0)&(df.L_RACE_OTH==0)).astype(int)
for col in ['L_RACE_AFR', 'L_RACE_ASI', 'L_RACE_HIS', 'L_RACE_White', 'L_RACE_OTH']:
    print(f'\t{col}, AFib+: {(df[col][ids1]==1).sum()} ({(df[col][ids1]==1).mean()*100:.1f}%)\tAFib-: {(df[col][ids0]==1).sum()} ({(df[col][ids0]==1).mean()*100:.1f}%)')

col = 'L_GIEDUC'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')

col = 'L_HWBMI'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')

col = 'M_AHI3'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')
ids2 = df[col]<5
print(f'\t0-5, AFib+: {len(df[ids1&ids2])} ({len(df[ids1&ids2])/len(df[ids1])*100:.1f}%)\tAFib-: {len(df[ids0&ids2])} ({len(df[ids0&ids2])/len(df[ids0])*100:.1f}%)')
ids2 = (df[col]>=5)&(df[col]<15)
print(f'\t5-15, AFib+: {len(df[ids1&ids2])} ({len(df[ids1&ids2])/len(df[ids1])*100:.1f}%)\tAFib-: {len(df[ids0&ids2])} ({len(df[ids0&ids2])/len(df[ids0])*100:.1f}%)')
ids2 = (df[col]>=15)&(df[col]<30)
print(f'\t15-30, AFib+: {len(df[ids1&ids2])} ({len(df[ids1&ids2])/len(df[ids1])*100:.1f}%)\tAFib-: {len(df[ids0&ids2])} ({len(df[ids0&ids2])/len(df[ids0])*100:.1f}%)')
ids2 = df[col]>=30
print(f'\t30-, AFib+: {len(df[ids1&ids2])} ({len(df[ids1&ids2])/len(df[ids1])*100:.1f}%)\tAFib-: {len(df[ids0&ids2])} ({len(df[ids0&ids2])/len(df[ids0])*100:.1f}%)')

col = 'L_EPEPWORT'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')

col = 'L_TURSMOKE'
print(col)
for vi, val in enumerate(['Never', 'Past', 'Current']):
    print(f'\t{val}, AFib+: {(df[col][ids1]==vi).sum()} ({(df[col][ids1]==vi).mean()*100:.1f}%)\tAFib-: {(df[col][ids0]==vi).sum()} ({(df[col][ids0]==vi).mean()*100:.1f}%)')

col = 'L_DPGDS15'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')

col = 'L_QLFXST51'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')

col = 'L_PASCORE'
print(f'{col}(AFib+) = {df[col][ids1].mean():.1f} ({df[col][ids1].std():.1f})\t{col}(AFib-) = {df[col][ids0].mean():.1f} ({df[col][ids0].std():.1f})')

print('Medical History')
for col in ['L_MHCHF', 'L_MHMI', 'L_MHBP', 'L_MHSTRK', 'L_MHDIAB']:
    print(f'\t{col}, AFib+: {(df[col][ids1]==1).sum()} ({(df[col][ids1]==1).mean()*100:.1f}%)\tAFib-: {(df[col][ids0]==1).sum()} ({(df[col][ids0]==1).mean()*100:.1f}%)')

print('Medication')
for col in ['L_AntiDep', 'L_Benzo']:
    print(f'\t{col}, AFib+: {(df[col][ids1]==1).sum()} ({(df[col][ids1]==1).mean()*100:.1f}%)\tAFib-: {(df[col][ids0]==1).sum()} ({(df[col][ids0]==1).mean()*100:.1f}%)')

col = 'L_VSCPAP'
print(f'{col}, AFib+: {(df[col][ids1]==1).sum()} ({(df[col][ids1]==1).mean()*100:.1f}%)\tAFib-: {(df[col][ids0]==1).sum()} ({(df[col][ids0]==1).mean()*100:.1f}%)')

