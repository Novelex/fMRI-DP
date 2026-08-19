"""Stage 8D arm runner — trained representation-health trajectory.

ONE scientific purpose: does the Stage-8C representation-health decline RECOVER,
STABILIZE or PERSIST, and is it specific to --phase_state_mode consistent or also
present under matched legacy training?

Calls the PRODUCTION unsupervised.training.train_one_epoch unchanged. No objective,
weight, data, Mij, TAE or loss change. Instrumentation is read-only:
  * torch.nn.utils.clip_grad_norm_ is wrapped to RECORD the pre-clip Phi gradient norm
    and to write the heartbeat -- it calls through to the real implementation, so
    behavior is identical.
  * all representation metrics are computed in a separate no_grad probe.

Usage (see EXACT_COMMAND_TEMPLATE.txt).
"""
import argparse, json, os, socket, subprocess, sys, time, hashlib

REPO = '/users/3171356m/muhammad/GraSTIACL'
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from datasets import ADNIDataset
import unsupervised.training as UT
from unsupervised.training import build_model_and_view_learner, train_one_epoch

for _m in (UT, sys.modules['datasets.Dataset']):
    assert _m.__file__.startswith(REPO), f'PROVENANCE FAIL {_m.__file__}'

BIAS = 1e-4
CKPT_EPOCHS = {0, 1, 3, 5, 10, 20, 30}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def erank(M):
    """Entropy effective rank of the mean-centred matrix (Stage-6 convention)."""
    M = M - M.mean(0, keepdim=True)
    s = torch.linalg.svdvals(M.float())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


def align_uniform(z, z_aug, t=2.0):
    """Wang & Isola (2020).
       alignment  = E_pos ||f(x) - f(x+)||^2          (lower = better aligned)
       uniformity = log E_{i!=j} exp(-t ||f(xi)-f(xj)||^2)  (lower = more uniform)
    Both on L2-NORMALISED embeddings, matching the paper's definition."""
    a, b = F.normalize(z, dim=1), F.normalize(z_aug, dim=1)
    alignment = float((a - b).pow(2).sum(1).mean())
    sq = torch.cdist(a, a).pow(2)
    n = a.shape[0]
    off = sq[~torch.eye(n, dtype=torch.bool, device=sq.device)]
    uniformity = float(torch.log(torch.exp(-t * off).mean()))
    return alignment, uniformity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--phase_state_mode', choices=['legacy', 'consistent'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--reg_lambda', type=float, required=True)
    ap.add_argument('--node_feature_mode', default='alff_new_z')
    # frozen Stage-8C configuration (NOT the argparse defaults of GraSTIACL.py)
    ap.add_argument('--ce_lambda', type=float, default=2.0)
    ap.add_argument('--kld_lambda', type=float, default=0.003)
    ap.add_argument('--model_lr', type=float, default=5e-4)
    ap.add_argument('--view_lr', type=float, default=5e-4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--emb_dim', type=int, default=32)
    ap.add_argument('--num_gc_layers', type=int, default=2)
    ap.add_argument('--vib_hidden_dim', type=int, default=64)
    ap.add_argument('--drop_ratio', type=float, default=0.0)
    ap.add_argument('--tae_profile', default='paper_intent')
    ap.add_argument('--mij_source', default='alff')
    ap.add_argument('--num_dyn_windows', type=int, default=3)
    ap.add_argument('--template', type=int, default=90)
    ap.add_argument('--stall_threshold', type=float, default=3600.0)
    args = ap.parse_args()

    od = args.outdir
    os.makedirs(f'{od}/checkpoints', exist_ok=True)
    log = open(f'{od}/train.log', 'a', buffering=1)

    def P(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True); log.write(msg + '\n')

    status = {'arm': args.arm, 'state': 'STARTING', 'epoch': 0,
              'started': time.time(), 'nonfinite': False, 'stalled': False,
              'valid_early_termination': None, 'reason': None}

    def write_status(**kw):
        status.update(kw); status['updated'] = time.time()
        json.dump(status, open(f'{od}/status.json', 'w'), indent=1)

    def heartbeat(epoch, batch, loss):
        json.dump({'timestamp': time.time(), 'iso': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'epoch': epoch, 'batch': batch, 'last_finite_loss': loss},
                  open(f'{od}/heartbeat', 'w'))

    # ---------------- provenance artefacts ----------------
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd=REPO).stdout.strip()
    open(f'{od}/git_commit.txt', 'w').write(commit + '\n')
    open(f'{od}/command.txt', 'w').write(' '.join([sys.executable] + sys.argv) + '\n')
    open(f'{od}/environment.txt', 'w').write(
        f'host={socket.gethostname()}\npython={sys.executable}\ntorch={torch.__version__}\n'
        f'cuda_available={torch.cuda.is_available()}\n'
        f'device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}\n'
        f'SLURM_JOB_ID={os.environ.get("SLURM_JOB_ID")}\n'
        f'SLURM_ARRAY_TASK_ID={os.environ.get("SLURM_ARRAY_TASK_ID")}\n')
    cfg = vars(args).copy()
    cfg.update(git_commit=commit,
               training_py_sha256=sha256(f'{REPO}/unsupervised/training.py'),
               dataset_py_sha256=sha256(f'{REPO}/datasets/Dataset.py'))
    json.dump(cfg, open(f'{od}/config.json', 'w'), indent=1)
    write_status(state='RUNNING')
    heartbeat(0, 0, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P(f'[{args.arm}] device={device} commit={commit[:8]} mode={args.phase_state_mode} '
      f'seed={args.seed} reg={args.reg_lambda} epochs={args.epochs}')

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = ADNIDataset('data/GraSTIACL_ABIDE_979', 'GraSTIACL_ABIDE_979',
                     node_feature_mode=args.node_feature_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model, vl, m_opt, v_opt, gamma_orig, beta = build_model_and_view_learner(
        num_dataset_features=3, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
        drop_ratio=args.drop_ratio, pooling_type='standard', gamma_mode='baseline',
        mij_source=args.mij_source, num_dyn_windows=args.num_dyn_windows,
        vib_hidden_dim=args.vib_hidden_dim, model_lr=args.model_lr, view_lr=args.view_lr,
        device=device, enable_attention_mix=True, signed_edges=True,
        tae_profile=args.tae_profile)

    # fixed probe cohort (deterministic, never shuffled)
    PROBE = [(k * 37) % len(ds) for k in range(96)]
    probe_loader = DataLoader([ds[i] for i in PROBE], batch_size=32, shuffle=False)

    # ---------------- read-only instrumentation of the clip call ----------------
    CLIP = {'fired': 0, 'steps': 0, 'norms': [], 'epoch': 0, 'batch': 0}
    _real_clip = torch.nn.utils.clip_grad_norm_

    def clip_hook(params, max_norm, *a, **kw):
        total = _real_clip(params, max_norm, *a, **kw)
        tn = float(total)
        CLIP['steps'] += 1; CLIP['batch'] += 1
        CLIP['norms'].append(tn)
        if tn > float(max_norm):
            CLIP['fired'] += 1
        heartbeat(CLIP['epoch'], CLIP['batch'], CLIP.get('last_loss'))
        return total
    torch.nn.utils.clip_grad_norm_ = clip_hook
    UT.torch.nn.utils.clip_grad_norm_ = clip_hook

    def probe(epoch):
        model.eval(); vl.eval()
        Z, ZA, acc = [], [], {k: [] for k in
                              ('CL', 'CE', 'KLD', 'REG', 'pos', 'mneg', 'gm', 'gs',
                               'align', 'unif', 'q25', 'q50', 'q75', 'top1')}
        with torch.no_grad():
            for b in probe_loader:
                b = b.to(device)
                gcn_w = b.edge_weight
                el, mu, std, ep = vl(b.batch, b.x, b.edge_index, beta, None, gcn_w,
                                     b.edge_weight, b.dyn_weight, gamma=gamma_orig)
                eps = (BIAS - (1 - BIAS)) * torch.rand(el.size(), device=device) + (1 - BIAS)
                gi = torch.log(eps) - torch.log(1 - eps)
                gate = torch.sigmoid((gi + el) / 1).squeeze()
                ga = gate.view(b.num_graphs, -1).mean(1)
                z, _ = model(b.batch, b.x, b.edge_index, beta, None, gcn_w, b.edge_weight,
                             None, gamma=gamma_orig)
                za, _ = model(b.batch, b.x, b.edge_index, beta, None, gcn_w * gate,
                              b.edge_weight, None, gamma=ga)
                acc['CL'].append(float(model.calc_loss(z, za)))
                a_, b_ = F.normalize(z, dim=1), F.normalize(za, dim=1)
                C = a_ @ b_.T; n = C.shape[0]
                acc['pos'].append(float(C.diag().mean()))
                acc['mneg'].append(float(((C.sum(1) - C.diag()) / (n - 1)).mean()))
                acc['top1'].append(float((C.argmax(1) == torch.arange(n, device=device)).float().mean()))
                al, un = align_uniform(z, za); acc['align'].append(al); acc['unif'].append(un)
                acc['gm'].append(float(gate.mean())); acc['gs'].append(float(gate.std()))
                q = torch.quantile(gate, torch.tensor([0.25, 0.5, 0.75], device=device))
                acc['q25'].append(float(q[0])); acc['q50'].append(float(q[1])); acc['q75'].append(float(q[2]))
                row, _ = b.edge_index; eb = b.batch[row]
                uni, cnt = eb.unique(return_counts=True); sp = scatter(1 - gate, eb, reduce='sum')
                acc['REG'].append(float(torch.stack(
                    [sp[i] / cnt[uni.tolist().index(i)] for i in range(b.num_graphs) if i in uni]).mean()))
                m2 = mu.reshape(b.num_graphs, -1); s2 = std.reshape(b.num_graphs, -1)
                acc['KLD'].append(float(-0.5 * torch.mean((1 + 2 * s2.log() - m2.pow(2) - s2.pow(2)).sum(1))))
                acc['CE'].append(float(F.binary_cross_entropy(
                    torch.sigmoid((el + gi) / 1), torch.sigmoid(ep.squeeze()).detach())))
                Z.append(z.cpu()); ZA.append(za.cpu())
        Z = torch.cat(Z); ZA = torch.cat(ZA)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        m.update(epoch=epoch, rank_z=erank(Z), rank_z_aug=erank(ZA),
                 margin=m['pos'] - m['mneg'],
                 J_Phi=-m['CL'] + args.ce_lambda * m['CE'] + args.kld_lambda * m['KLD'],
                 finite=bool(torch.isfinite(Z).all() and torch.isfinite(ZA).all()),
                 min_norm_z=float(Z.norm(dim=1).min()), min_norm_z_aug=float(ZA.norm(dim=1).min()),
                 norm_ratio=float(ZA.norm(dim=1).mean() / Z.norm(dim=1).mean()))
        return m

    def save_ckpt(ep):
        torch.save({'epoch': ep, 'arm': args.arm, 'git_commit': commit,
                    'phase_state_mode': args.phase_state_mode, 'seed': args.seed,
                    'reg_lambda': args.reg_lambda,
                    'model': model.state_dict(), 'view_learner': vl.state_dict(),
                    'model_optimizer': m_opt.state_dict(),
                    'view_optimizer': v_opt.state_dict(),
                    'beta': beta, 'gamma_orig': gamma_orig},
                   f'{od}/checkpoints/epoch{ep}.pt')

    METRICS = []
    t0 = time.time()
    m0 = probe(0); METRICS.append(m0); save_ckpt(0)
    P(f'  epoch0 CL={m0["CL"]:.4f} rank_z={m0["rank_z"]:.2f} rank_z_aug={m0["rank_z_aug"]:.2f} '
      f'pos={m0["pos"]:.4f} margin={m0["margin"]:+.5f} align={m0["align"]:.4f} unif={m0["unif"]:.4f}')
    json.dump(METRICS, open(f'{od}/metrics.json', 'w'), indent=1)

    last_hb = time.time()
    for ep in range(1, args.epochs + 1):
        CLIP['epoch'] = ep; CLIP['batch'] = 0
        te = time.time()
        try:
            out = train_one_epoch(model, vl, m_opt, v_opt, loader, device, beta, gamma_orig,
                                  ce_lambda=args.ce_lambda, reg_lambda=args.reg_lambda,
                                  kld_lambda=args.kld_lambda, template=args.template,
                                  epoch_num=ep, signed_edges=True,
                                  phase_state_mode=args.phase_state_mode)
        except Exception as e:
            P(f'  EPOCH {ep} RAISED: {type(e).__name__}: {e}')
            write_status(state='FAILED', epoch=ep, reason=f'{type(e).__name__}: {e}')
            sys.exit(3)
        CLIP['last_loss'] = out[0]
        dt = time.time() - te
        if not (np.isfinite(out[0]) and np.isfinite(out[1])):
            P(f'  NON-FINITE objective at epoch {ep}: model={out[0]} view={out[1]}')
            write_status(state='NONFINITE', epoch=ep, nonfinite=True,
                         reason='non-finite training objective')
            save_ckpt(ep); sys.exit(4)
        heartbeat(ep, CLIP['batch'], out[0])
        m = probe(ep)
        m.update(train_model_loss=out[0], train_view_loss=out[1], train_reg=out[2],
                 seconds=dt, clip_fired=CLIP['fired'], clip_steps=CLIP['steps'],
                 clip_pct=100.0 * CLIP['fired'] / max(CLIP['steps'], 1),
                 phi_grad_norm_preclip_mean=float(np.mean(CLIP['norms'][-len(loader):]))
                 if CLIP['norms'] else None)
        METRICS.append(m)
        json.dump(METRICS, open(f'{od}/metrics.json', 'w'), indent=1)
        P(f'  epoch{ep} CL={m["CL"]:.4f} J={m["J_Phi"]:.3f} rank_z={m["rank_z"]:.2f} '
          f'rank_z_aug={m["rank_z_aug"]:.2f} pos={m["pos"]:.4f} margin={m["margin"]:+.5f} '
          f'align={m["align"]:.4f} unif={m["unif"]:.4f} gate={m["gm"]:.4f}+-{m["gs"]:.4f} '
          f'|gPhi|={m["phi_grad_norm_preclip_mean"]:.3f} clip={m["clip_pct"]:.1f}% ({dt:.0f}s)')
        if not m['finite']:
            P(f'  NON-FINITE embedding at epoch {ep}')
            write_status(state='NONFINITE', epoch=ep, nonfinite=True,
                         reason='non-finite embedding'); save_ckpt(ep); sys.exit(4)
        if ep in CKPT_EPOCHS or ep == args.epochs:
            save_ckpt(ep)
        write_status(state='RUNNING', epoch=ep)

    write_status(state='COMPLETED', epoch=args.epochs,
                 seconds_total=time.time() - t0)
    P(f'[{args.arm}] COMPLETED {args.epochs} epochs in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
