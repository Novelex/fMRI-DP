import os,sys,json
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from torch_geometric.data import Batch
from datasets import ADNIDataset
from unsupervised.training import (build_model_and_view_learner, _bn_buffer_snapshot,
                                   _bn_buffer_restore, _pair_gammas)
ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode='alff_new_z')
IDX=[(k*37)%len(ds) for k in range(96)]
ARMS=['arm01_consistent_seed42','arm02_consistent_seed7','arm03_consistent_seed2024',
      'arm04_legacy_seed42','arm05_legacy_seed7','arm06_legacy_seed2024']
def load(arm,ep):
    cfg=json.load(open(f'stage8d/{arm}/config.json'))
    ck=torch.load(f'stage8d/{arm}/checkpoints/epoch{ep}.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(cfg['seed'])
    m,v,_,_,g,b=build_model_and_view_learner(num_dataset_features=3,emb_dim=32,num_gc_layers=2,
        drop_ratio=0.0,pooling_type='standard',gamma_mode='baseline',mij_source='alff',num_dyn_windows=3,
        vib_hidden_dim=64,model_lr=5e-4,view_lr=5e-4,device=torch.device('cpu'),
        enable_attention_mix=True,signed_edges=True,tae_profile='paper_intent')
    m.load_state_dict(ck['model']); v.load_state_dict(ck['view_learner']); return m,v,g,b
def measure(m,v,g0,beta,train_state,pairing):
    bn_m,bn_v=_bn_buffer_snapshot(m),_bn_buffer_snapshot(v)
    m.train(train_state); v.train(train_state); T1=[];PR=[];CL=[]
    torch.manual_seed(4242)
    with torch.no_grad():
        for s in range(0,96,32):
            b=Batch.from_data_list([ds[i] for i in IDX[s:s+32]]); w=b.edge_weight
            el,_,_,_=v(b.batch,b.x,b.edge_index,beta,None,w,b.edge_weight,b.dyn_weight,gamma=g0)
            e=(1e-4-(1-1e-4))*torch.rand(el.size())+(1-1e-4)
            gate=torch.sigmoid((torch.log(e)-torch.log(1-e)+el.squeeze())/1)
            ga=gate.view(32,-1).mean(1)
            go,gaa=_pair_gammas(pairing,g0,ga)
            z,_=m(b.batch,b.x,b.edge_index,beta,None,w,b.edge_weight,None,gamma=go)
            za,_=m(b.batch,b.x,b.edge_index,beta,None,w*gate,b.edge_weight,None,gamma=gaa)
            C=F.normalize(z,dim=1)@F.normalize(za,dim=1).T
            T1.append(float((C.argmax(1)==torch.arange(32)).float().mean()))
            PR.append(float((C>C.diag().unsqueeze(1)).sum(1).float().mean()+1))
            CL.append(float(m.calc_loss(z,za)))
    _bn_buffer_restore(m,bn_m); _bn_buffer_restore(v,bn_v)
    return np.mean(T1),np.mean(PR),np.mean(CL)
PAIR=['production','matched','balanced','attention_off']
print("=== top1 (null 0.03125) / posRank (null 16.5), epoch 30, frozen weights ===")
print(f"{'arm':26s} {'state':6s} "+" ".join(f"{p[:9]:>17s}" for p in PAIR))
for arm in ARMS:
    m,v,g0,beta=load(arm,30)
    for st,nm in ((False,'EVAL'),(True,'TRAIN')):
        cells=[]
        for p in PAIR:
            t1,pr,cl=measure(m,v,g0,beta,st,p); cells.append(f"{t1:6.4f}/{pr:5.2f}")
        print(f"{arm:26s} {nm:6s} "+" ".join(f"{c:>17s}" for c in cells))
print("\n=== BatchNorm-only control: EVAL state, production pairing, but BN buffers")
print("    replaced by the probe cohort's own batch statistics (i.e. BN gap removed,")
print("    lambda mismatch KEPT) ===")
for arm in ARMS:
    m,v,g0,beta=load(arm,30)
    e_t1,e_pr,_=measure(m,v,g0,beta,False,'production')
    t_t1,t_pr,_=measure(m,v,g0,beta,True,'production')
    em_t1,em_pr,_=measure(m,v,g0,beta,False,'matched')
    tm_t1,tm_pr,_=measure(m,v,g0,beta,True,'matched')
    print(f"{arm:26s} eval/prod {e_t1:.4f}  ->remove BN gap (train/prod) {t_t1:.4f}"
          f"  ->remove lam gap (eval/matched) {em_t1:.4f}  ->remove BOTH {tm_t1:.4f}")
