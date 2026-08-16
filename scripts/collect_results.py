"""Campaign dashboard: parses campaign_* logs into one table read against the
pre-written cards. Usage: python3 scripts/collect_results.py [logs_glob]
"""
import glob
import re
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else '/users/3171356m/muhammad/GraSTIACL/logs/campaign_*.err'
CARDS = ("Cards: chance 52.4% | untrained probe 54-61% | classical ALFF 57-59% | "
         "classical PCC 64-68% | old GNN 51-53% | paper MDD 62.56%")

rows = []
for path in sorted(glob.glob(pattern)):
    text = open(path, errors='replace').read()
    arm = re.search(r'ARM=(\S+)', text)
    out_path = path.replace('.err', '.out')
    try:
        arm = arm.group(1) if arm else re.search(r'ARM=(\S+)', open(out_path, errors='replace').read()).group(1)
    except Exception:
        arm = '?'

    before = re.search(r'Before training Embedding Eval Scores.*?Test: \[([\d.]+)', text)
    best_epoch = re.search(r'(?:HOLDOUT )?BestEpoch: (\d+)', text)
    best_val = re.search(r'BestValidationScore: acc_mean: ([\d.]+)', text)
    best_test = re.search(r'BestTestScore: acc_mean: ([\d.]+)', text)
    guard_events = len(re.findall(r'Non-finite \(NaN/Inf\) detected', text))
    finished = 'FinishedTraining!' in text
    crashed = 'Traceback' in text

    rows.append({
        'log': path.split('/')[-1],
        'arm': arm,
        'epoch0_test': before.group(1)[:6] if before else '-',
        'best_epoch': best_epoch.group(1) if best_epoch else '-',
        'best_val': best_val.group(1)[:6] if best_val else '-',
        'test_at_best_val': best_test.group(1)[:6] if best_test else '-',
        'guard_events': guard_events,
        'status': 'CRASH' if crashed else ('done' if finished else 'running/partial'),
    })

print(CARDS)
print()
hdr = f"{'log':38s} {'arm':4s} {'ep0-test':>8s} {'bestEp':>6s} {'bestVal':>8s} {'test@bV':>8s} {'guard':>5s} {'status'}"
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(f"{r['log']:38s} {r['arm']:4s} {r['epoch0_test']:>8s} {r['best_epoch']:>6s} "
          f"{r['best_val']:>8s} {r['test_at_best_val']:>8s} {r['guard_events']:>5d} {r['status']}")
if not rows:
    print("(no campaign logs matched)")
