"""Stage 8D postflight validator. Runs with afterany so it reports failures too."""
import json, os, glob, subprocess, sys
REPO='/users/3171356m/muhammad/GraSTIACL'; os.chdir(REPO)
MANDATORY=[f'arm0{i}' for i in range(1,7)]
rows=[]
for d in sorted(glob.glob('stage8d/arm*')):
    name=os.path.basename(d)
    cfg=json.load(open(f'{d}/config.json')) if os.path.exists(f'{d}/config.json') else {}
    st=json.load(open(f'{d}/status.json')) if os.path.exists(f'{d}/status.json') else {}
    met=json.load(open(f'{d}/metrics.json')) if os.path.exists(f'{d}/metrics.json') else []
    cks=sorted(int(os.path.basename(p)[5:-3]) for p in glob.glob(f'{d}/checkpoints/epoch*.pt'))
    exp=cfg.get('epochs'); last=met[-1]['epoch'] if met else None
    state=st.get('state','MISSING')
    ok = (state=='COMPLETED' and last==exp)
    special = state in ('NONFINITE',) and cfg.get('reg_lambda') in (0.0,1.0)
    rows.append(dict(ARM=name, SCHEDULER_STATE=state, EXPECTED_EPOCHS=exp,
        LAST_COMPLETED_EPOCH=last, CHECKPOINTS_PRESENT=cks,
        NONFINITE=st.get('nonfinite'), STALLED=st.get('stalled'),
        VALID_EARLY_TERMINATION=bool(special),
        PHASE_STATE_MODE=cfg.get('phase_state_mode'), SEED=cfg.get('seed'),
        REG=cfg.get('reg_lambda'), CONFIG_GIT=cfg.get('git_commit','')[:8],
        RESULT='PASS' if ok else ('VALID_SPECIAL_RESULT' if special else 'FAIL')))
complete = all(any(r['ARM'].startswith(m) and r['RESULT']=='PASS' for r in rows) for m in MANDATORY)
out=['STAGE8D POSTFLIGHT', f'STAGE8D_ARRAY_COMPLETE = {"YES" if complete else "NO"}','']
hdr=f"{'ARM':30s} {'STATE':12s} {'EXP':>4s} {'LAST':>5s} {'MODE':11s} {'SEED':>5s} {'REG':>5s} {'CKPTS':22s} {'RESULT'}"
out.append(hdr); out.append('-'*len(hdr))
for r in rows:
    out.append(f"{r['ARM']:30s} {str(r['SCHEDULER_STATE']):12s} {str(r['EXPECTED_EPOCHS']):>4s} "
               f"{str(r['LAST_COMPLETED_EPOCH']):>5s} {str(r['PHASE_STATE_MODE']):11s} "
               f"{str(r['SEED']):>5s} {str(r['REG']):>5s} {str(r['CHECKPOINTS_PRESENT'])[:22]:22s} {r['RESULT']}")
# Section 17: config-divergence check between arm01 and arm04 (same seed, mode differs only)
try:
    m1=json.load(open('stage8d/arm01_consistent_seed42/metrics.json'))
    m4=json.load(open('stage8d/arm04_legacy_seed42/metrics.json'))
    c1=json.load(open('stage8d/arm01_consistent_seed42/config.json'))
    c4=json.load(open('stage8d/arm04_legacy_seed42/config.json'))
    e1=[x for x in m1 if x['epoch']==1]; e4=[x for x in m4 if x['epoch']==1]
    if e1 and e4:
        a,b=e1[0]['CL'],e4[0]['CL']
        out += ['', 'SECTION 17 CONFIG-DIVERGENCE CHECK (arm01 vs arm04, seed 42)',
                f"  arm01 saved mode = {c1['phase_state_mode']}   arm04 saved mode = {c4['phase_state_mode']}",
                f"  arm1 epoch1 CL = {a:.6f}", f"  arm4 epoch1 CL = {b:.6f}",
                f"  |diff| = {abs(a-b):.3e}",
                f"  PHASE_STATE_MODE_THREADING_CONCERN = "
                f"{'YES (modes differ but trajectories implausibly identical)' if abs(a-b)<1e-6 else 'NO'}"]
except Exception as e:
    out += ['', f'SECTION 17 check unavailable: {e}']
txt='\n'.join(out)
open('stage8d/POSTFLIGHT_SUMMARY.txt','w').write(txt+'\n')
json.dump(dict(complete=complete, arms=rows), open('stage8d/POSTFLIGHT_SUMMARY.json','w'), indent=1)
print(txt)
