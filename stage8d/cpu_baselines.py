"""Stage 8D CPU baselines -- classical references on the SAME 956 alff_new_z cohort.

LinearSVC on: (a) static FC upper triangle, (b) flattened ALFF, (c) FC + ALFF.
5 folds x 3 seeds, scaling fit on TRAIN ONLY, inner train-only C selection. No leakage.
Cohort is recorded exactly; 954 and 956 are never mixed.
"""
import os, sys, json, numpy as np
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from datasets import ADNIDataset

MODE='alff_new_z'
ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode=MODE)
N=len(ds); print(f'cohort: {MODE}  N={N}')
iu=np.triu_indices(90,k=1)
FC=np.zeros((N,len(iu[0],)),dtype=np.float32); AL=np.zeros((N,270),dtype=np.float32); y=np.zeros(N,dtype=int)
for i in range(N):
    d=ds[i]
    FC[i]=d.edge_weight.view(90,90).numpy()[iu]
    AL[i]=d.x.numpy().ravel()
    y[i]=int(d.y.view(-1)[0])
FEATS={'FC_uppertri':FC,'ALFF_flat':AL,'FC_plus_ALFF':np.concatenate([FC,AL],1)}
GRID={'linearsvc__C':[0.001,0.01,0.1,1.0,10.0]}
res=[]
for name,X in FEATS.items():
    for seed in (0,1,2):
        skf=StratifiedKFold(5,shuffle=True,random_state=seed)
        ba,au=[],[]
        for tr,te in skf.split(X,y):
            pipe=make_pipeline(StandardScaler(),LinearSVC(dual=False,max_iter=5000))
            gs=GridSearchCV(pipe,GRID,cv=StratifiedKFold(3,shuffle=True,random_state=seed),
                            scoring='balanced_accuracy',n_jobs=4)
            gs.fit(X[tr],y[tr])                      # train-only scaling + train-only C selection
            p=gs.predict(X[te]); s=gs.decision_function(X[te])
            ba.append(balanced_accuracy_score(y[te],p)); au.append(roc_auc_score(y[te],s))
        res.append(dict(feature=name,seed=seed,bal_acc=float(np.mean(ba)),auc=float(np.mean(au)),
                        bal_acc_sd=float(np.std(ba))))
        print(f'  {name:14s} seed{seed}  balAcc={np.mean(ba):.4f}+-{np.std(ba):.4f}  AUC={np.mean(au):.4f}',flush=True)
summary={'cohort':MODE,'N':int(N),'n_asd':int((y==1).sum()),'n_nc':int((y==0).sum()),'results':res}
for name in FEATS:
    r=[x['bal_acc'] for x in res if x['feature']==name]
    summary[f'{name}_mean_balacc']=float(np.mean(r))
    print(f'{name:14s} MEAN balAcc over 3 seeds = {np.mean(r):.4f}')
json.dump(summary,open('stage8d/cpu_baselines.json','w'),indent=1)
print('wrote stage8d/cpu_baselines.json')
