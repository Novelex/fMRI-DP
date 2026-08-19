import os,sys,json
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from torch_geometric.data import Batch
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner
BIAS=1e-4
ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode='alff_new_z')
def load(arm,ep):
    cfg=json.load(open(f'stage8d/{arm}/config.json'))
    ck=torch.load(f'stage8d/{arm}/checkpoints/epoch{ep}.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(cfg['seed'])
    m,v,mo,vo,g,b=build_model_and_view_learner(num_dataset_features=3,emb_dim=32,num_gc_layers=2,
        drop_ratio=0.0,pooling_type='standard',gamma_mode='baseline',mij_source='alff',num_dyn_windows=3,
        vib_hidden_dim=64,model_lr=5e-4,view_lr=5e-4,device=torch.device('cpu'),
        enable_attention_mix=True,signed_edges=True,tae_profile='paper_intent')
    m.load_state_dict(ck['model']); v.load_state_dict(ck['view_learner']); return m,v,g,b
m,v,g0,beta=load('arm01_consistent_seed42',30)
IDX=[(k*9)%len(ds) for k in range(96)]
b=Batch.from_data_list([ds[i] for i in IDX[:32]])
v.eval()
with torch.no_grad():
    torch.manual_seed(777)
    el,_,_,_=v(b.batch,b.x,b.edge_index,beta,None,b.edge_weight,b.edge_weight,b.dyn_weight,gamma=g0)
    torch.manual_seed(778)
    e=(BIAS-(1-BIAS))*torch.rand(el.size())+(1-BIAS)
    gate=torch.sigmoid((torch.log(e)-torch.log(1-e)+el.squeeze())/1)
    ga=gate.view(32,-1).mean(1)
print("=== SECTION 6: lambda distributions, paper_intent (gamma_orig=1.0), epoch30 ===")
def lam_stats(gam,mode,N=128):
    gs=torch.as_tensor(gam,dtype=torch.float32).clamp(1e-4,1-1e-4)
    if mode=='eval': L=(1-gs).repeat(N,1) if gs.dim() else (1-gs).repeat(N)
    else:
        L=torch.stack([torch.distributions.Beta(1-gs,gs).sample() for _ in range(N)])
    L=L.flatten().numpy()
    q=np.quantile(L,[0.01,0.05,0.25,0.5,0.75,0.95,0.99])
    return L.mean(),L.std(),q,(L<0.05).mean(),(L>0.95).mean()
for label,gam in (('ORIGINAL (gamma=1.0)',torch.tensor(1.0)),('AUGMENTED (gamma=mean gate)',ga)):
    for mode in ('eval','train'):
        mn,sd,q,plo,phi=lam_stats(gam,mode)
        print(f"  {label:28s} {mode:5s}: mean {mn:.5f} sd {sd:.5f} "
              f"q01/05/25/50/75/95/99 {'/'.join(f'{x:.3f}' for x in q)}  P(<.05)={plo:.3f} P(>.95)={phi:.3f}")
print(f"  gamma_aug: mean {float(ga.mean()):.4f} sd {float(ga.std()):.4f}")
# branch norms
with torch.no_grad():
    enc=m.encoder; m.eval()
    xa=enc.trans_conv(b.x,b.batch)
    r2=F.relu(enc.bns[0](enc.convs[0](b.x,b.edge_index,b.edge_weight)))
    xt=enc.bns[1](enc.convs[1](r2,b.edge_index,b.edge_weight))
    r2a=F.relu(enc.bns[0](enc.convs[0](b.x,b.edge_index,b.edge_weight*gate)))
    xta=enc.bns[1](enc.convs[1](r2a,b.edge_index,b.edge_weight*gate))
def er(T):
    T=T-T.mean(0,keepdim=True); s=torch.linalg.svdvals(T.float()); s=s[s>1e-12]
    return float(torch.exp(-(p*torch.log(p)).sum())) if (p:=s/s.sum()).numel() else 0.
lo,la=1e-4,float((1-ga.clamp(1e-4,1-1e-4)).mean())
print(f"\n  ORIG : ||X_topo||={float(xt.norm(dim=1).mean()):.4f}  ||lam*X_atte||={lo*float(xa.norm(dim=1).mean()):.6f}  "
      f"ratio={lo*float(xa.norm(dim=1).mean())/float(xt.norm(dim=1).mean()):.6f}  "
      f"cos(topo,atte)={float(F.cosine_similarity(xt.flatten(),xa.flatten(),dim=0)):.4f}")
print(f"  AUG  : ||X_topo||={float(xta.norm(dim=1).mean()):.4f}  ||lam*X_atte||={la*float(xa.norm(dim=1).mean()):.4f}  "
      f"ratio={la*float(xa.norm(dim=1).mean())/float(xta.norm(dim=1).mean()):.4f}")
print(f"  eRank X_topo(orig)={er(xt):.2f}  X_atte={er(xa):.2f}  fused(orig)={er(xt+lo*xa):.2f}  fused(aug)={er(xta+la*xa):.2f}")

print("\n=== SECTION 7: positive-pair compatibility under lambda variation (hardest-negative aware) ===")
from torch_geometric.nn import global_add_pool
def embed(gam_o,gam_a):
    with torch.no_grad():
        z,_=m(b.batch,b.x,b.edge_index,beta,None,b.edge_weight,b.edge_weight,None,gamma=gam_o)
        za,_=m(b.batch,b.x,b.edge_index,beta,None,b.edge_weight*gate,b.edge_weight,None,gamma=gam_a)
    return z,za
def stats(z,za,tag):
    a_,b_=F.normalize(z,dim=1),F.normalize(za,dim=1); C=a_@b_.T; n=C.shape[0]
    pos=C.diag(); off=C.clone(); off.fill_diagonal_(-9e9)
    maxneg=off.max(1).values; mneg=(C.sum(1)-pos)/(n-1)
    rank=(C>pos.unsqueeze(1)).sum(1)+1
    lse=torch.logsumexp(off.masked_fill(off<-1e8,-1e9)/0.2,dim=1)
    print(f"  {tag:34s} pos {float(pos.mean()):.4f}  maxneg {float(maxneg.mean()):.4f}  "
          f"mneg {float(mneg.mean()):.4f}  pos-maxneg {float((pos-maxneg).mean()):+.4f}  "
          f"posRank {float(rank.float().mean()):.2f}/{n}  LSEneg {float(lse.mean()):.3f}  "
          f"top1 {float((rank==1).float().mean()):.3f}")
    return float((pos-maxneg).mean()), float(rank.float().mean())
gmid=float(ga.mean())
r={}
r['A']=stats(*embed(1.0,ga),'A mismatched (orig g=1, aug g=mean)')
r['B']=stats(*embed(gmid,ga),'B shared gamma=mean(gate)')
r['C']=stats(*embed(1.0,torch.full_like(ga,1.0)),'C shared gamma=1 (attention off both)')
r['D']=stats(*embed(0.5,torch.full_like(ga,0.5)),'D shared gamma=0.5 (lam=0.5 both)')
brk = (r['A'][1] > r['B'][1]*1.5) or (r['A'][0] < r['B'][0]-0.05)
print(f"\n  LAMBDA_MISMATCH_BREAKS_POSITIVE_IDENTITY = {'YES' if brk else ('PARTIAL' if r['A'][1]>r['B'][1] else 'NO')}")
