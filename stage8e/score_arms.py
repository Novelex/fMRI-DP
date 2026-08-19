"""Stage 8E — score every completed arm against stage8e/PREREGISTERED_CRITERIA.md.

Reads only metrics.json / status.json. Applies the four pre-registered primary
criteria, expressed relative to each arm's OWN InfoNCE null so that E5 (B=128) is
comparable to the B=32 arms. Prints the cross-pairing table that separates a
representation rescue from a measurement-only rescue.
"""
import json, os, sys, glob
import numpy as np

REPO = '/users/3171356m/muhammad/GraSTIACL'
OUT = f'{REPO}/stage8e'

def crit(m, nl):
    """The four pre-registered criteria, on the eval/own surface, vs this arm's null."""
    return dict(
        top1   = (m['top1']   >= 8.0 * nl['null_top1'],   m['top1'],   8.0 * nl['null_top1']),
        posRank= (m['posRank'] <= nl['probe_batch_size'] / 4.0, m['posRank'], nl['probe_batch_size'] / 4.0),
        CL     = (m['CL']     <= nl['null_CL'] - 0.2657,  m['CL'],     nl['null_CL'] - 0.2657),
        rank   = (m['rank_z'] >= 4.0,                     m['rank_z'], 4.0))

rows = []
for d in sorted(glob.glob(f'{OUT}/E*/') + glob.glob(f'{OUT}/R*/')):
    name = os.path.basename(d.rstrip('/'))
    if not os.path.exists(f'{d}/metrics.json'):
        rows.append((name, 'NO_METRICS', None, None)); continue
    M = json.load(open(f'{d}/metrics.json'))
    st = json.load(open(f'{d}/status.json')) if os.path.exists(f'{d}/status.json') else {}
    cfg = json.load(open(f'{d}/config.json'))
    last = M[-1]
    state = st.get('state', '?')
    if state == 'NONFINITE': verdict = 'NON_FINITE'
    elif state != 'COMPLETED': verdict = f"INCOMPLETE_{last['epoch']}"
    else:
        c = crit(last['eval_own'], last)
        verdict = 'RESCUED' if all(v[0] for v in c.values()) else (
            'PARTIAL' if any(v[0] for v in c.values()) else 'NOT_RESCUED')
    rows.append((name, verdict, last, cfg))

print(f"{'arm':22s} {'pairing':14s} {'B':>4s} {'ep':>3s} {'verdict':16s}")
print('-' * 66)
for name, v, last, cfg in rows:
    if last is None: print(f"{name:22s} {'-':14s} {'-':>4s} {'-':>3s} {v:16s}"); continue
    print(f"{name:22s} {cfg['lambda_pairing_mode']:14s} {cfg['batch_size']:4d} "
          f"{last['epoch']:3d} {v:16s}")

print(f"\n=== PRIMARY criteria on eval/own (null-relative) ===")
print(f"{'arm':22s} {'top1':>18s} {'posRank':>18s} {'CL':>18s} {'rank_z':>14s}")
for name, v, last, cfg in rows:
    if last is None or 'eval_own' not in last: continue
    c = crit(last['eval_own'], last)
    f = lambda k, fmt: f"{c[k][1]:{fmt}}{'PASS' if c[k][0] else 'fail':>6s}({c[k][2]:{fmt}})"
    print(f"{name:22s} {f('top1','.4f'):>18s} {f('posRank','.2f'):>18s} "
          f"{f('CL','.3f'):>18s} {f('rank','.2f'):>14s}")

print(f"\n=== CROSS-PAIRING / CROSS-STATE table (epoch 30) ===")
print(f"{'arm':22s} {'surface':12s} {'CL':>8s} {'top1':>8s} {'posRank':>9s} {'rank_z':>8s} "
      f"{'margin':>9s} {'unif':>8s} {'lam_o':>7s} {'lam_a':>7s}")
for name, v, last, cfg in rows:
    if last is None: continue
    for s in ('eval_own', 'eval_prod', 'train_own', 'train_prod'):
        if s not in last: continue
        m = last[s]
        print(f"{name:22s} {s:12s} {m['CL']:8.4f} {m['top1']:8.4f} {m['posRank']:9.2f} "
              f"{m['rank_z']:8.2f} {m['margin']:+9.5f} {m['unif']:8.3f} "
              f"{m['lam_orig']:7.4f} {m['lam_aug']:7.4f}")

# representation vs measurement rescue (pre-declared rule)
E0 = next((r for r in rows if r[0].startswith('E0') and r[2] and 'eval_prod' in r[2]), None)
if E0:
    base = E0[2]['eval_prod']
    print(f"\n=== RESCUE CLASSIFICATION (vs E0 eval_prod top1={base['top1']:.4f}, "
          f"posRank={base['posRank']:.2f}) ===")
    for name, v, last, cfg in rows:
        if last is None or v not in ('RESCUED', 'PARTIAL') or name.startswith('E0'): continue
        beats = last['eval_prod']['top1'] > base['top1'] and last['eval_prod']['posRank'] < base['posRank']
        print(f"  {name:22s} {v:10s} -> "
              f"{'RESCUED_REPRESENTATION' if (v=='RESCUED' and beats) else ('RESCUED_MEASUREMENT_ONLY' if v=='RESCUED' else 'PARTIAL')} "
              f"(eval_prod top1 {last['eval_prod']['top1']:.4f}, posRank {last['eval_prod']['posRank']:.2f})")

resc = [r[0] for r in rows if r[1] == 'RESCUED']
print(f"\nRESCUED_ARMS = {resc if resc else 'NONE'}")
if not resc:
    print("VERDICT = NO_ARM_RESCUES (pre-registered: no replication is run)")
else:
    best = max((r for r in rows if r[1] == 'RESCUED'),
               key=lambda r: (r[2]['eval_own']['top1'], -r[2]['eval_own']['CL']))
    print(f"REPLICATE = {best[0]}  (pairing={best[3]['lambda_pairing_mode']})")
json.dump([(n, v, (c or {}).get('lambda_pairing_mode')) for n, v, _, c in rows],
          open(f'{OUT}/scoreboard.json', 'w'), indent=1)
