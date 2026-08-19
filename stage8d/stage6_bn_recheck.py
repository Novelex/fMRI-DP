"""Stage 8D section 15 -- rerun the Stage-6 representation pathway at epoch-0 weights
using TRAIN-LIKE BatchNorm statistics. Stage 6 measured everything in eval mode while the
running buffers were still mean=0/var=1, which Stage 8C showed is a degenerate regime.
Read-only. Does NOT reopen ALFF."""
import os, sys, json
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import numpy as np, torch, torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner

def erank(M):
    M=M-M.mean(0,keepdim=True); s=torch.linalg.svdvals(M.float()); s=s[s>1e-12]
    if s.numel()==0: return 0.0
    p=s/s.sum(); return float(torch.exp(-(p*torch.log(p)).sum()))

ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode='alff_new_z')
torch.manual_seed(42)
m,v,_,_,g0,beta=build_model_and_view_learner(
    num_dataset_features=3,emb_dim=32,num_gc_layers=2,drop_ratio=0.0,pooling_type='standard',
    gamma_mode='baseline',mij_source='alff',num_dyn_windows=3,vib_hidden_dim=64,
    model_lr=5e-4,view_lr=5e-4,device=torch.device('cpu'),enable_attention_mix=True,
    signed_edges=True,tae_profile='paper_intent')
enc=m.encoder
COH=[(k*9)%len(ds) for k in range(96)]
iu=torch.triu_indices(90,90,offset=1)
OUT={}
for bnmode in ('eval','train'):
    enc.train(False); enc.trans_conv.train(False)      # set the module tree FIRST ...
    for mod in enc.bns: mod.train(bnmode=='train')      # ... THEN override the BN flags
    assert enc.bns[0].training == (bnmode=='train'), 'BN flag did not take effect'
    R0,R2,R3,R6,R7,R8=[],[],[],[],[],[]
    cos0,cos2,cos3=[],[],[]
    with torch.no_grad():
        for s in range(0,96,32):
            b=Batch.from_data_list([ds[i] for i in COH[s:s+32]])
            w=b.edge_weight
            x=b.x
            xa=enc.trans_conv(x,b.batch)
            r2=F.relu(enc.bns[0](enc.convs[0](x,b.edge_index,w)))
            r3=enc.bns[1](enc.convs[1](r2,b.edge_index,w))
            lam=(1-torch.tensor(1.0).clamp(1e-4,1-1e-4))
            r6=r3+lam*xa
            r7=global_add_pool(r6,b.batch); r8=m.proj_head(r7)
            n=b.num_graphs
            for k in range(n):
                X0=x.view(n,90,-1)[k]; X2=r2.view(n,90,-1)[k]; X3=r3.view(n,90,-1)[k]
                R0.append(erank(X0)); R2.append(erank(X2)); R3.append(erank(X3))
                for arr,C in ((cos0,X0),(cos2,X2),(cos3,X3)):
                    Xn=F.normalize(C,dim=1); arr.append(float((Xn@Xn.T)[iu[0],iu[1]].mean()))
            R6.append(r6); R7.append(r7); R8.append(r8)
    R7=torch.cat(R7); R8=torch.cat(R8)
    OUT[bnmode]=dict(R0_erank=float(np.mean(R0)),R2_erank=float(np.mean(R2)),R3_erank=float(np.mean(R3)),
        R0_roicos=float(np.mean(cos0)),R2_roicos=float(np.mean(cos2)),R3_roicos=float(np.mean(cos3)),
        R7_subject_erank=erank(R7),R8_subject_erank=erank(R8),
        R7_norm=float(R7.norm(dim=1).mean()),R8_norm=float(R8.norm(dim=1).mean()))
    print(f"BN={bnmode:5s} R0eR={OUT[bnmode]['R0_erank']:.2f} R2eR={OUT[bnmode]['R2_erank']:.2f} "
          f"R3eR={OUT[bnmode]['R3_erank']:.2f} ROIcos R0/R2/R3="
          f"{OUT[bnmode]['R0_roicos']:.4f}/{OUT[bnmode]['R2_roicos']:.4f}/{OUT[bnmode]['R3_roicos']:.4f} "
          f"subjR7={OUT[bnmode]['R7_subject_erank']:.2f} subjR8={OUT[bnmode]['R8_subject_erank']:.2f}",flush=True)
e,t=OUT['eval'],OUT['train']
verdict='PARTIAL'
if t['R2_erank']>2*e['R2_erank'] and t['R7_subject_erank']>2*e['R7_subject_erank']: verdict='NO'
elif abs(t['R2_erank']-e['R2_erank'])<0.5 and abs(t['R7_subject_erank']-e['R7_subject_erank'])<0.5: verdict='YES'
OUT['STAGE6_FINDINGS_SURVIVE_TRAIN_MODE_BN']=verdict
print("STAGE6_FINDINGS_SURVIVE_TRAIN_MODE_BN =",verdict)
json.dump(OUT,open('stage8d/stage6_bn_recheck.json','w'),indent=1)
