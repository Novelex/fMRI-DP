import os,sys,json
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner
from unsupervised.convs.gcn_conv import gcn_norm
ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode='alff_new_z')
IDX=[(k*37)%len(ds) for k in range(96)]
ARMS=['arm01_consistent_seed42','arm02_consistent_seed7','arm03_consistent_seed2024',
      'arm04_legacy_seed42','arm05_legacy_seed7','arm06_legacy_seed2024']
def er(T):
    T=T-T.mean(0,keepdim=True); s=torch.linalg.svdvals(T.float()); s=s[s>1e-12]
    return float(torch.exp(-(p*torch.log(p)).sum())) if (p:=s/s.sum()).numel() else 0.
def load(arm,ep):
    cfg=json.load(open(f'stage8d/{arm}/config.json'))
    ck=torch.load(f'stage8d/{arm}/checkpoints/epoch{ep}.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(cfg['seed'])
    m,v,_,_,g,b=build_model_and_view_learner(num_dataset_features=3,emb_dim=32,num_gc_layers=2,
        drop_ratio=0.0,pooling_type='standard',gamma_mode='baseline',mij_source='alff',num_dyn_windows=3,
        vib_hidden_dim=64,model_lr=5e-4,view_lr=5e-4,device=torch.device('cpu'),
        enable_attention_mix=True,signed_edges=True,tae_profile='paper_intent')
    m.load_state_dict(ck['model']); v.load_state_dict(ck['view_learner']); return m,v,g,b
def stages(m,beta,g0):
    enc=m.encoder; m.eval(); acc={k:[] for k in ('R0','GCN1','GCN2','XATTE','FUSION','POOL','PROJ')}
    with torch.no_grad():
        for s in range(0,96,32):
            b=Batch.from_data_list([ds[i] for i in IDX[s:s+32]]); n=b.num_graphs; w=b.edge_weight
            xa=enc.trans_conv(b.x,b.batch)
            r2=F.relu(enc.bns[0](enc.convs[0](b.x,b.edge_index,w)))
            r3=enc.bns[1](enc.convs[1](r2,b.edge_index,w))
            fu=r3+1e-4*xa; pool=global_add_pool(fu,b.batch)
            for k,T in (('R0',b.x),('GCN1',r2),('GCN2',r3),('XATTE',xa),('FUSION',fu)):
                acc[k].append(T.view(n,90,-1).mean(1))
            acc['POOL'].append(pool); acc['PROJ'].append(m.proj_head(pool))
    return {k:er(torch.cat(v)) for k,v in acc.items()}
print("=== SECTION 8: layer localization, all six main arms ===")
print(f"{'arm':26s} {'ep':>3s} {'R0':>6s} {'GCN1':>6s} {'GCN2':>6s} {'XATTE':>6s} {'FUSION':>7s} {'POOL':>6s} {'PROJ':>6s}")
FIRST={}
for arm in ARMS:
    ref=None
    for ep in (0,10,30):
        s=stages(*load(arm,ep)[0:1]+ (load(arm,ep)[3], load(arm,ep)[2]))
        print(f"{arm:26s} {ep:3d} "+" ".join(f"{s[k]:6.2f}" for k in ('R0','GCN1','GCN2','XATTE','FUSION','POOL','PROJ')))
        if ep==0: ref=s
        if ep==30:
            first=None
            for k in ('R0','GCN1','GCN2','XATTE','FUSION','POOL','PROJ'):
                if s[k]-ref[k] < -1.0: first=k; break
            FIRST[arm]=first or 'NONE'
print("\nFIRST_COMPRESSION_STAGE_PER_ARM:")
for a,f in FIRST.items(): print(f"  {a:26s} {f}")
g2=sum(1 for f in FIRST.values() if f=='GCN2')
print(f"GCN2_COMPRESSION_REPLICATES = {'YES' if g2>=5 else ('PARTIAL' if g2>=2 else 'NO')}  ({g2}/6 arms)")

print("\n=== SECTION 10: spectral diagnostics of the SIGNED-SAFE propagation operator ===")
m,v,g0,beta=load('arm01_consistent_seed42',30)
b=Batch.from_data_list([ds[i] for i in IDX[:8]])
with torch.no_grad():
    ei,coef=gcn_norm(b.edge_index,b.edge_weight.clone(),b.num_nodes,False,False,signed_safe=True)
    S=torch.zeros(b.num_nodes,b.num_nodes); S[ei[0],ei[1]]=coef
    S1=S[:90,:90]
    sv=torch.linalg.svdvals(S1); ev=torch.linalg.eigvals(S1).abs()
    X=b.x[:90]
    def erm(T):
        s=torch.linalg.svdvals(T.float()); s=s[s>1e-12]; p=s/s.sum()
        return float(torch.exp(-(p*torch.log(p)).sum()))
    print(f"  spectral radius (|eig|max) {float(ev.max()):.4f}  2nd |eig| {float(ev.sort(descending=True).values[1]):.4f}  "
          f"gap {float(ev.max()-ev.sort(descending=True).values[1]):.4f}")
    print(f"  top singular values {[round(float(x),4) for x in sv[:5]]}")
    print(f"  eRank(S)={erm(S1):.2f}  eRank(S@S)={erm(S1@S1):.2f}  eRank(S@X)={erm(S1@X):.2f}  eRank(S^2@X)={erm(S1@S1@X):.2f}  eRank(X)={erm(X):.2f}")
    sup = erm(S1@S1@X) < erm(S1@X) < erm(X)
    print(f"  DENSE_GCN_OVERSMOOTHING_SUPPORTED = {'YES' if sup else 'PARTIAL'}  (rank must fall monotonically with propagation depth)")

print("\n=== SECTION 11: collapsed-state escape gradients (epoch30, consistent seeds) ===")
print(f"{'arm':26s} {'|g GCN1|':>9s} {'|g BN1|':>9s} {'|g GCN2|':>9s} {'|g BN2|':>9s} {'|g attn|':>9s} {'|g proj|':>9s}")
for arm in ARMS[:3]:
    m,v,g0,beta=load(arm,30); m.train(); v.eval()
    b=Batch.from_data_list([ds[i] for i in IDX[:32]])
    torch.manual_seed(4242)
    with torch.no_grad():
        el,_,_,_=v(b.batch,b.x,b.edge_index,beta,None,b.edge_weight,b.edge_weight,b.dyn_weight,gamma=g0)
        e=(1e-4-(1-1e-4))*torch.rand(el.size())+(1-1e-4)
        gate=torch.sigmoid((torch.log(e)-torch.log(1-e)+el.squeeze())/1); ga=gate.view(32,-1).mean(1)
    z,_=m(b.batch,b.x,b.edge_index,beta,None,b.edge_weight,b.edge_weight,None,gamma=g0)
    za,_=m(b.batch,b.x,b.edge_index,beta,None,b.edge_weight*gate,b.edge_weight,None,gamma=ga)
    L=m.calc_loss(z,za); m.zero_grad(); L.backward()
    def gn(mod): return float(sum((p.grad.pow(2).sum() for p in mod.parameters() if p.grad is not None),torch.tensor(0.)).sqrt())
    print(f"{arm:26s} {gn(m.encoder.convs[0]):9.4f} {gn(m.encoder.bns[0]):9.4f} {gn(m.encoder.convs[1]):9.4f} "
          f"{gn(m.encoder.bns[1]):9.4f} {gn(m.encoder.trans_conv):9.4f} {gn(m.proj_head):9.4f}")
