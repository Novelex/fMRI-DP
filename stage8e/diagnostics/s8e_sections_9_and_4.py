import os,sys,json
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO); sys.path.insert(0,REPO)
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner
ds=ADNIDataset('data/GraSTIACL_ABIDE_979','GraSTIACL_ABIDE_979',node_feature_mode='alff_new_z')
IDX=[(k*37)%len(ds) for k in range(96)]
def er(T):
    T=T-T.mean(0,keepdim=True); s=torch.linalg.svdvals(T.float()); s=s[s>1e-12]
    if s.numel()==0: return 0.
    p=s/s.sum(); return float(torch.exp(-(p*torch.log(p)).sum()))
def load(arm,ep):
    cfg=json.load(open(f'stage8d/{arm}/config.json'))
    ck=torch.load(f'stage8d/{arm}/checkpoints/epoch{ep}.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(cfg['seed'])
    m,v,_,_,g,b=build_model_and_view_learner(num_dataset_features=3,emb_dim=32,num_gc_layers=2,
        drop_ratio=0.0,pooling_type='standard',gamma_mode='baseline',mij_source='alff',num_dyn_windows=3,
        vib_hidden_dim=64,model_lr=5e-4,view_lr=5e-4,device=torch.device('cpu'),
        enable_attention_mix=True,signed_edges=True,tae_profile='paper_intent')
    m.load_state_dict(ck['model']); v.load_state_dict(ck['view_learner']); return m,v,g,b

print("=== SECTION 9: GCN2 decomposed -- feature transform vs propagation vs BatchNorm ===")
print("(subject-level entropy effective rank, 96 subjects, eval state)")
print(f"{'arm':26s} {'ep':>3s} {'in(r2)':>7s} {'W@r2':>7s} {'S@W@r2':>7s} {'+bias':>7s} {'BN2':>7s} "
      f"{'dW':>6s} {'dS':>6s} {'dBN':>6s}")
ROWS=[]
for arm in ['arm01_consistent_seed42','arm02_consistent_seed7','arm03_consistent_seed2024',
            'arm04_legacy_seed42','arm05_legacy_seed7','arm06_legacy_seed2024']:
    for ep in (0,30):
        m,v,g0,beta=load(arm,ep); enc=m.encoder; m.eval()
        A={k:[] for k in ('in','W','SW','SWb','BN')}
        with torch.no_grad():
            for s in range(0,96,32):
                b=Batch.from_data_list([ds[i] for i in IDX[s:s+32]]); n=b.num_graphs; w=b.edge_weight
                r2=F.relu(enc.bns[0](enc.convs[0](b.x,b.edge_index,w)))
                c=enc.convs[1]
                ei,ew=c._cached_edge_index if c._cached_edge_index is not None else \
                    __import__('unsupervised.convs.gcn_conv',fromlist=['gcn_norm']).gcn_norm(
                        b.edge_index,w.clone(),r2.size(0),c.improved,c.add_self_loops,signed_safe=c.signed_safe)
                Wr=c.lin(r2)
                SWr=c.propagate(ei,x=Wr,edge_weight=ew,size=None)
                SWb=SWr+c.bias if c.bias is not None else SWr
                BN=enc.bns[1](SWb)
                for k,T in (('in',r2),('W',Wr),('SW',SWr),('SWb',SWb),('BN',BN)):
                    A[k].append(T.view(n,90,-1).mean(1))
        R={k:er(torch.cat(vv)) for k,vv in A.items()}
        print(f"{arm:26s} {ep:3d} {R['in']:7.2f} {R['W']:7.2f} {R['SW']:7.2f} {R['SWb']:7.2f} {R['BN']:7.2f} "
              f"{R['W']-R['in']:+6.2f} {R['SW']-R['W']:+6.2f} {R['BN']-R['SWb']:+6.2f}")
        ROWS.append((arm,ep,R))
d={'W':[],'S':[],'BN':[]}
for arm,ep,R in ROWS:
    if ep==30: d['W'].append(R['W']-R['in']); d['S'].append(R['SW']-R['W']); d['BN'].append(R['BN']-R['SWb'])
print(f"\nepoch30 mean rank change: feature-transform {np.mean(d['W']):+.2f}  "
      f"propagation {np.mean(d['S']):+.2f}  BatchNorm {np.mean(d['BN']):+.2f}")
dom = max((('FEATURE_TRANSFORM',np.mean(d['W'])),('PROPAGATION',np.mean(d['S'])),('BATCHNORM',np.mean(d['BN']))),key=lambda t:-t[1])
print(f"GCN2_RANK_LOSS_DOMINATED_BY = {dom[0]} ({dom[1]:+.2f})")

print("\n=== SECTION 4 (completion): Phi gradient decomposition into view.net ===")
m,v,g0,beta=load('arm01_consistent_seed42',0); m0,v0=m,v
for ep,(m,v,_,_) in (( 0,load('arm01_consistent_seed42',0)),(30,load('arm01_consistent_seed42',30))):
    b=Batch.from_data_list([ds[i] for i in IDX[:32]])
    m.eval(); v.train()
    gcn_w=b.edge_weight
    def terms():
        el,mu,std,eprod=v(b.batch,b.x,b.edge_index,beta,None,gcn_w,b.edge_weight,b.dyn_weight,gamma=g0)
        torch.manual_seed(7)
        eps=(1e-4-(1-1e-4))*torch.rand(el.size())+(1-1e-4)
        gi=torch.log(eps)-torch.log(1-eps); gate=torch.sigmoid((gi+el)/1).squeeze()
        ga=gate.view(32,-1).mean(1)
        with torch.no_grad(): z,_=m(b.batch,b.x,b.edge_index,beta,None,gcn_w,b.edge_weight,None,gamma=g0)
        za,_=m(b.batch,b.x,b.edge_index,beta,None,gcn_w*gate,b.edge_weight,None,gamma=ga)
        CL=m.calc_loss(z,za)
        CE=F.binary_cross_entropy(torch.sigmoid((el+gi)/1),torch.sigmoid(eprod.squeeze()).detach())
        m2=mu.reshape(32,-1); s2=std.reshape(32,-1)
        KLD=-0.5*torch.mean((1+2*s2.log()-m2.pow(2)-s2.pow(2)).sum(1))
        from torch_scatter import scatter
        row,_=b.edge_index; eb=b.batch[row]
        uni,cnt=eb.unique(return_counts=True); sp=scatter(1-gate,eb,reduce="sum")
        REG=torch.stack([sp[i]/cnt[uni.tolist().index(i)] for i in range(32) if i in uni]).mean()
        return {'CL':CL,'CE':CE,'KLD':KLD,'REG':REG}
    T=terms()
    P=[p for n,p in v.named_parameters() if n.startswith('net.')]
    G={}
    for k,val in T.items():
        v.zero_grad()
        gs=torch.autograd.grad(val,P,retain_graph=True,allow_unused=True)
        G[k]=torch.cat([ (g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,P)])
    W={'CL':1.0,'CE':-2.0,'KLD':-0.003,'REG':-0.2}   # dJ_Phi/dPhi for view_loss = CL - 2CE - .003KLD - .2REG
    tot=sum(W[k]*G[k] for k in G)
    print(f"\n  epoch{ep}: ||g|| by term (into view.net, {sum(p.numel() for p in P)} params)")
    for k in ('CL','CE','KLD','REG'):
        print(f"    {k:4s} raw {float(G[k].norm()):10.4f}  weighted {float((W[k]*G[k]).norm()):10.4f}  "
              f"share {100*float((W[k]*G[k]).norm())/sum(float((W[j]*G[j]).norm()) for j in G):5.1f}%  "
              f"cos(term, total) {float(F.cosine_similarity((W[k]*G[k]).unsqueeze(0),tot.unsqueeze(0))):+.4f}")
    print(f"    TOTAL ||g|| {float(tot.norm()):.4f}")
    print("    pairwise cosines:")
    ks=['CL','CE','KLD','REG']
    for i in range(4):
        for j in range(i+1,4):
            print(f"      cos({ks[i]:>4s},{ks[j]:>4s}) = "
                  f"{float(F.cosine_similarity(G[ks[i]].unsqueeze(0),G[ks[j]].unsqueeze(0))):+.4f}")
