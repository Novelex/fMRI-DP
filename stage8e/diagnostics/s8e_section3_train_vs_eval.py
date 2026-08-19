import os,sys,json
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from torch_geometric.data import Batch
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner, _bn_buffer_snapshot, _bn_buffer_restore
ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode='alff_new_z')
IDX=[(k*37)%len(ds) for k in range(96)]
def load(arm,ep):
    cfg=json.load(open(f'stage8d/{arm}/config.json'))
    ck=torch.load(f'stage8d/{arm}/checkpoints/epoch{ep}.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(cfg['seed'])
    m,v,_,_,g,b=build_model_and_view_learner(num_dataset_features=3,emb_dim=32,num_gc_layers=2,
        drop_ratio=0.0,pooling_type='standard',gamma_mode='baseline',mij_source='alff',num_dyn_windows=3,
        vib_hidden_dim=64,model_lr=5e-4,view_lr=5e-4,device=torch.device('cpu'),
        enable_attention_mix=True,signed_edges=True,tae_profile='paper_intent')
    m.load_state_dict(ck['model']); v.load_state_dict(ck['view_learner']); return m,v,g,b
print("TRAIN-mode vs EVAL-mode contrastive loss at epoch 30 (all six Stage-8D main arms)")
print(f"{'arm':26s} {'TRAIN CL':>9s} {'TRAIN top1':>11s} {'TRAIN posR':>11s} | {'EVAL CL':>8s} {'EVAL top1':>10s} {'EVAL posR':>10s}")
for arm in ['arm01_consistent_seed42','arm02_consistent_seed7','arm03_consistent_seed2024',
            'arm04_legacy_seed42','arm05_legacy_seed7','arm06_legacy_seed2024']:
    m,v,g0,beta=load(arm,30); res={}
    for tr,nm in ((True,'train'),(False,'eval')):
        bn_m,bn_v=_bn_buffer_snapshot(m),_bn_buffer_snapshot(v)
        m.train(tr); v.train(tr); CL=[];T1=[];PR=[]
        torch.manual_seed(4242)
        with torch.no_grad():
            for s in range(0,96,32):
                b=Batch.from_data_list([ds[i] for i in IDX[s:s+32]]); w=b.edge_weight
                el,_,_,_=v(b.batch,b.x,b.edge_index,beta,None,w,b.edge_weight,b.dyn_weight,gamma=g0)
                e=(1e-4-(1-1e-4))*torch.rand(el.size())+(1-1e-4)
                gate=torch.sigmoid((torch.log(e)-torch.log(1-e)+el.squeeze())/1)
                ga=gate.view(32,-1).mean(1)
                z,_=m(b.batch,b.x,b.edge_index,beta,None,w,b.edge_weight,None,gamma=g0)
                za,_=m(b.batch,b.x,b.edge_index,beta,None,w*gate,b.edge_weight,None,gamma=ga)
                CL.append(float(m.calc_loss(z,za)))
                C=F.normalize(z,dim=1)@F.normalize(za,dim=1).T
                T1.append(float((C.argmax(1)==torch.arange(32)).float().mean()))
                PR.append(float((C>C.diag().unsqueeze(1)).sum(1).float().mean()+1))
        _bn_buffer_restore(m,bn_m); _bn_buffer_restore(v,bn_v)
        res[nm]=(np.mean(CL),np.mean(T1),np.mean(PR))
    t,e=res['train'],res['eval']
    print(f"{arm:26s} {t[0]:9.4f} {t[1]:11.4f} {t[2]:11.2f} | {e[0]:8.4f} {e[1]:10.4f} {e[2]:10.2f}")
print("\nnull: CL=3.4657 top1=0.03125 posRank=16.5")
