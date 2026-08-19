"""Stage 8E — score every arm against the SIX governing rescue criteria.

Criteria (fixed before any arm ran; see stage8e/PREREGISTERED_CRITERIA.md and
docs/STAGE8E_COLLAPSE_MECHANISM_AUDIT.md section 14). Compared with E0:

  C1 sustained improvement in CL_EXCESS (= CL - log B) at >= 2 LATE checkpoints
  C2 positive-minus-HARDEST-negative improves materially
  C3 positive rank improves
  C4 uniformity stays meaningfully below 0 rather than collapsing to ~0
  C5 subject effective rank does not collapse to the E0 regime
  C6 the effect is NOT a one-checkpoint transient

Rescue is never declared from positive cosine alone. "Late" = epochs >= 20.
All criteria are read on the eval surface, under BOTH the arm's own pairing and the
fixed 'production' pairing, because E1-E3 change the ruler as well as the encoder.
"""
import json, os, glob, sys
import numpy as np

OUT = '/users/3171356m/muhammad/GraSTIACL/stage8e'
LATE = 20                    # epochs >= LATE are "late checkpoints"
MATERIAL_PMH = 0.02          # C2: material improvement in pos - hardest-neg
MATERIAL_CLX = 0.05          # C1: material improvement in CL excess (nats)
UNIF_MEANINGFUL = -0.10      # C4: uniformity must stay below this
RANK_FACTOR = 1.25           # C5: rank must exceed 1.25x the E0 regime


def load():
    arms = {}
    for d in sorted(glob.glob(f'{OUT}/E*/') + glob.glob(f'{OUT}/R*/')):
        n = os.path.basename(d.rstrip('/'))
        if not os.path.exists(f'{d}/metrics.json'):
            arms[n] = dict(state='NO_METRICS'); continue
        arms[n] = dict(M=json.load(open(f'{d}/metrics.json')),
                       cfg=json.load(open(f'{d}/config.json')),
                       state=json.load(open(f'{d}/status.json')).get('state', '?')
                       if os.path.exists(f'{d}/status.json') else '?')
    return arms


def series(M, surface, key):
    return [(m['epoch'], m[surface][key]) for m in M if surface in m]


def excess(M, surface):
    return [(m['epoch'], m[surface]['CL'] - m['null_CL']) for m in M if surface in m]


def judge(a, e0, surface):
    """Apply C1..C6. Returns (verdict, per-criterion dict)."""
    M, M0 = a['M'], e0['M']
    lastN = lambda s: [v for ep, v in s if ep >= LATE]
    ax, bx = lastN(excess(M, surface)), lastN(excess(M0, surface))
    b_cx = np.mean(bx) if bx else 0.0
    c1n = sum(1 for v in ax if v < b_cx - MATERIAL_CLX)
    C1 = (c1n >= 2, f"{c1n}/{len(ax)} late ckpts improve CL_excess by >{MATERIAL_CLX} "
                    f"(arm {np.mean(ax):+.3f} vs E0 {b_cx:+.3f})")

    ap, bp = lastN(series(M, surface, 'pos_minus_maxneg')), lastN(series(M0, surface, 'pos_minus_maxneg'))
    d2 = np.mean(ap) - np.mean(bp) if ap and bp else 0.0
    C2 = (d2 > MATERIAL_PMH, f"pos-hardestneg {np.mean(ap):+.4f} vs E0 {np.mean(bp):+.4f} (d {d2:+.4f})")

    ar, br = lastN(series(M, surface, 'posRank')), lastN(series(M0, surface, 'posRank'))
    C3 = (np.mean(ar) < np.mean(br) - 0.5, f"posRank {np.mean(ar):.2f} vs E0 {np.mean(br):.2f}")

    au = lastN(series(M, surface, 'unif'))
    C4 = (np.mean(au) < UNIF_MEANINGFUL, f"uniformity {np.mean(au):+.4f} (needs < {UNIF_MEANINGFUL})")

    ak, bk = lastN(series(M, surface, 'rank_z')), lastN(series(M0, surface, 'rank_z'))
    C5 = (np.mean(ak) > RANK_FACTOR * np.mean(bk), f"rank_z {np.mean(ak):.2f} vs E0 regime {np.mean(bk):.2f}")

    # C6: the improvement is present at more than one late checkpoint on the two
    # scale-free identity measures, i.e. not a single-epoch spike.
    at = lastN(series(M, surface, 'top1')); bt = lastN(series(M0, surface, 'top1'))
    nsp = sum(1 for v in at if v > np.mean(bt) * 2 + 1e-9)
    C6 = (nsp >= 2, f"{nsp}/{len(at)} late ckpts hold >2x E0 top1 ({np.mean(bt):.4f})")

    C = dict(C1=C1, C2=C2, C3=C3, C4=C4, C5=C5, C6=C6)
    npass = sum(v[0] for v in C.values())
    verdict = 'RESCUE' if npass == 6 else ('PARTIAL' if npass >= 3 else 'NO_RESCUE')
    return verdict, C, npass


def main():
    arms = load()
    if 'E0_baseline' not in arms or 'M' not in arms.get('E0_baseline', {}):
        print('E0_baseline has no metrics yet -- cannot score.'); return
    e0 = arms['E0_baseline']

    print('=== arm status ===')
    for n, a in arms.items():
        if 'M' not in a: print(f'  {n:22s} {a["state"]}'); continue
        print(f'  {n:22s} {a["state"]:12s} last epoch {a["M"][-1]["epoch"]:3d} '
              f'pairing={a["cfg"]["lambda_pairing_mode"]:14s} B={a["cfg"]["batch_size"]}')

    results = {}
    for surface in ('eval_own', 'eval_prod'):
        print(f'\n=== SIX GOVERNING CRITERIA on {surface} (late = epochs >= {LATE}) ===')
        for n, a in arms.items():
            if 'M' not in a or n == 'E0_baseline': continue
            if a['state'] != 'COMPLETED':
                print(f'  {n:22s} {a["state"]} -- not scored'); continue
            v, C, npass = judge(a, e0, surface)
            results.setdefault(n, {})[surface] = (v, npass)
            print(f'  {n:22s} {v:10s} ({npass}/6)')
            for k in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
                print(f'      [{"PASS" if C[k][0] else "fail"}] {k}: {C[k][1]}')

    print('\n=== late-epoch panel (mean over epochs >= 20) ===')
    hdr = f"{'arm':22s} {'surface':10s} {'CLexcess':>9s} {'top1':>8s} {'posRank':>8s} " \
          f"{'p-hneg':>8s} {'unif':>8s} {'rank_z':>7s}"
    print(hdr)
    for n, a in arms.items():
        if 'M' not in a: continue
        for surface in ('eval_own', 'eval_prod', 'train_own'):
            L = [m for m in a['M'] if m['epoch'] >= LATE and surface in m]
            if not L: continue
            f = lambda k: np.mean([m[surface][k] for m in L])
            cx = np.mean([m[surface]['CL'] - m['null_CL'] for m in L])
            print(f"{n:22s} {surface:10s} {cx:+9.4f} {f('top1'):8.4f} {f('posRank'):8.2f} "
                  f"{f('pos_minus_maxneg'):+8.4f} {f('unif'):+8.4f} {f('rank_z'):7.2f}")

    resc = [n for n, r in results.items() if r.get('eval_own', ('', 0))[0] == 'RESCUE']
    print(f"\nRESCUED_ARMS (eval_own) = {resc if resc else 'NONE'}")
    if resc:
        best = max(resc, key=lambda n: np.mean(
            [m['eval_own']['top1'] for m in arms[n]['M'] if m['epoch'] >= LATE]))
        also = results[best].get('eval_prod', ('', 0))[0]
        print(f"BEST_MECHANISTIC_ARM = {best} (pairing={arms[best]['cfg']['lambda_pairing_mode']})")
        print(f"  on the common yardstick eval_prod it scores: {also}")
        print(f"  -> {'RESCUED_REPRESENTATION' if also == 'RESCUE' else 'RESCUED_ENCODING_PAIRING'}")
    json.dump({n: {k: v[0] for k, v in r.items()} for n, r in results.items()},
              open(f'{OUT}/scoreboard.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
