"""Stage 8E arm runner (derived from stage8d/train_arm.py) — trained representation-health trajectory.

ONE scientific purpose: run the pre-registered Stage-8E causal arms (E0-E5), each
changing exactly ONE factor, and decide against stage8e/PREREGISTERED_CRITERIA.md
whether any of them rescues same-subject identity.

Two measurement changes vs the Stage-8D runner, both forced by Stage-8E findings:
  * every metric is probed in BOTH model.train() and model.eval() state, because
    Stage 8E Section 3 showed those differ radically (train CL descends to ~0.13
    while eval CL sits at ~3.44) -- an eval-only probe cannot tell an optimiser
    failure from a train/eval transfer gap.
  * every metric is probed under BOTH the arm's OWN view pairing and the fixed
    'production' pairing, because arms E1-E3 change how the positive pair is
    encoded and would otherwise appear to improve purely by changing the ruler.
    posRank is added as a scale-free identity measure (null = (B+1)/2).

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
    ap.add_argument('--lambda_pairing_mode', required=True,
                    choices=['production', 'matched', 'attention_off', 'balanced'])
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
               preregistered_criteria_sha256=sha256(f'{REPO}/stage8e/PREREGISTERED_CRITERIA.md'),
               training_py_sha256=sha256(f'{REPO}/unsupervised/training.py'),
               dataset_py_sha256=sha256(f'{REPO}/datasets/Dataset.py'))
    json.dump(cfg, open(f'{od}/config.json', 'w'), indent=1)
    write_status(state='RUNNING')
    heartbeat(0, 0, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P(f'[{args.arm}] device={device} commit={commit[:8]} state={args.phase_state_mode} '
      f'pairing={args.lambda_pairing_mode} seed={args.seed} reg={args.reg_lambda} '
      f'bs={args.batch_size} kld={args.kld_lambda} epochs={args.epochs}')

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
    # Probe cohort. The InfoNCE null depends on the batch size, so an arm that trains
    # at B=128 is probed at B=128 against its OWN null (log 128, chance top1 1/128).
    PROBE_B = args.batch_size
    PROBE_N = 96 if PROBE_B <= 32 else 3 * PROBE_B
    PROBE = [(k * 37) % len(ds) for k in range(PROBE_N)]
    probe_loader = DataLoader([ds[i] for i in PROBE], batch_size=PROBE_B, shuffle=False)

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

    from unsupervised.training import _pair_gammas

    def _probe_once(train_state, pairing):
        """One full pass over the fixed probe cohort in ONE forward state under ONE
        view pairing. No gradients, no optimiser, no buffer writes that survive:
        BatchNorm buffers are snapshotted and restored around a train-state probe so
        the probe can never perturb the run it is measuring."""
        from unsupervised.training import _bn_buffer_snapshot, _bn_buffer_restore
        bn_m, bn_v = _bn_buffer_snapshot(model), _bn_buffer_snapshot(vl)
        model.train(train_state); vl.train(train_state)
        Z, ZA, acc = [], [], {k: [] for k in
                              ('CL', 'CE', 'KLD', 'REG', 'pos', 'mneg', 'maxneg', 'gm', 'gs',
                               'align', 'unif', 'q25', 'q50', 'q75', 'top1', 'posRank',
                               'lam_orig', 'lam_aug')}
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
                g_o, g_a = _pair_gammas(pairing, gamma_orig, ga)
                z, _ = model(b.batch, b.x, b.edge_index, beta, None, gcn_w, b.edge_weight,
                             None, gamma=g_o)
                za, _ = model(b.batch, b.x, b.edge_index, beta, None, gcn_w * gate,
                              b.edge_weight, None, gamma=g_a)
                acc['CL'].append(float(model.calc_loss(z, za)))
                a_, b_ = F.normalize(z, dim=1), F.normalize(za, dim=1)
                C = a_ @ b_.T; n = C.shape[0]
                eye = torch.eye(n, dtype=torch.bool, device=C.device)
                acc['pos'].append(float(C.diag().mean()))
                acc['mneg'].append(float(((C.sum(1) - C.diag()) / (n - 1)).mean()))
                acc['maxneg'].append(float(C.masked_fill(eye, -2).max(1).values.mean()))
                acc['top1'].append(float((C.argmax(1) == torch.arange(n, device=device)).float().mean()))
                # rank of the positive within its own row, 1 = best (scale-free; null (B+1)/2)
                acc['posRank'].append(float((C > C.diag().unsqueeze(1)).sum(1).float().mean() + 1))
                # the two views' realised Eq.18 lambda under this pairing
                LM = lambda g: float((1 - torch.as_tensor(
                    g, dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)).mean())
                acc['lam_orig'].append(LM(g_o)); acc['lam_aug'].append(LM(g_a))
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
        _bn_buffer_restore(model, bn_m); _bn_buffer_restore(vl, bn_v)
        model.eval(); vl.eval()
        Z = torch.cat(Z); ZA = torch.cat(ZA)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        m.update(rank_z=erank(Z), rank_z_aug=erank(ZA), margin=m['pos'] - m['mneg'],
                 pos_minus_maxneg=m['pos'] - m['maxneg'],
                 J_Phi=-m['CL'] + args.ce_lambda * m['CE'] + args.kld_lambda * m['KLD'],
                 finite=bool(torch.isfinite(Z).all() and torch.isfinite(ZA).all()),
                 min_norm_z=float(Z.norm(dim=1).min()), min_norm_z_aug=float(ZA.norm(dim=1).min()),
                 norm_ratio=float(ZA.norm(dim=1).mean() / Z.norm(dim=1).mean()))
        return m

    def probe(epoch):
        """Four measurements per epoch: {eval, train} x {arm's own pairing, production}.
        'eval/own' is the pre-registered PRIMARY criterion surface; 'eval/production'
        is the common yardstick that makes arms comparable on one fixed ruler."""
        out = {'epoch': epoch, 'probe_batch_size': PROBE_B,
               'null_CL': float(np.log(PROBE_B)), 'null_top1': 1.0 / PROBE_B,
               'null_posRank': (PROBE_B + 1) / 2.0}
        for state, sname in ((False, 'eval'), (True, 'train')):
            for pairing, pname in ((args.lambda_pairing_mode, 'own'), ('production', 'prod')):
                if pname == 'prod' and args.lambda_pairing_mode == 'production':
                    out[f'{sname}_prod'] = out[f'{sname}_own']   # identical by construction
                    continue
                out[f'{sname}_{pname}'] = _probe_once(state, pairing)
        # flat aliases so downstream Stage-8D-style tooling keeps working; these are
        # explicitly the EVAL / OWN-pairing numbers, never a silent mixture.
        for k, v in out['eval_own'].items():
            out[k] = v
        return out

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
    P(f'  epoch0 EVAL CL={m0["eval_own"]["CL"]:.4f} top1={m0["eval_own"]["top1"]:.4f} '
      f'posRank={m0["eval_own"]["posRank"]:.2f}/{m0["null_posRank"]:.1f} '
      f'rank_z={m0["eval_own"]["rank_z"]:.2f} | TRAIN CL={m0["train_own"]["CL"]:.4f} '
      f'top1={m0["train_own"]["top1"]:.4f}')
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
                                  phase_state_mode=args.phase_state_mode,
                                  lambda_pairing_mode=args.lambda_pairing_mode)
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
        e_, t_ = m['eval_own'], m['train_own']
        P(f'  epoch{ep} EVAL CL={e_["CL"]:.4f} top1={e_["top1"]:.4f} '
          f'posRank={e_["posRank"]:.2f} rank_z={e_["rank_z"]:.2f} margin={e_["margin"]:+.5f} '
          f'unif={e_["unif"]:.4f} | TRAIN CL={t_["CL"]:.4f} top1={t_["top1"]:.4f} '
          f'posRank={t_["posRank"]:.2f} | lam {e_["lam_orig"]:.4f}/{e_["lam_aug"]:.4f} '
          f'gate={e_["gm"]:.4f}+-{e_["gs"]:.4f} |gPhi|={m["phi_grad_norm_preclip_mean"]:.3f} '
          f'clip={m["clip_pct"]:.1f}% ({dt:.0f}s)')
        if not m['eval_own']['finite']:
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
