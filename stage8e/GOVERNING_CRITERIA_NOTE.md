# Which criteria govern, and why there are two files

Two sets of rescue criteria exist. **Both predate the first arm launch**, so neither
is post-hoc, but they are not interchangeable and the precedence is fixed here.

1. **The Stage-8E specification, Section 15** — issued before any Stage-8E work began.
   Six criteria: sustained CL_EXCESS improvement at ≥2 late checkpoints;
   positive-minus-*hardest*-negative improves materially; positive rank improves;
   uniformity stays meaningfully below 0; subject effective rank does not collapse to
   the E0 regime; the effect is not a one-checkpoint transient. Plus: never declare
   rescue from positive cosine alone, and a locked LinearSVC probe as *secondary*
   evidence at epochs 0/10/30 on identical folds.

   **These are the GOVERNING criteria.** `stage8e/score_arms.py` implements exactly
   these six and reports each one's pass/fail with the numbers behind it.

2. **`stage8e/PREREGISTERED_CRITERIA.md`** (committed `f709a14`, sha256
   `34e0da03…babacd6`) — my own operationalisation, written to disk before the first
   arm launched. It fixes the *measurement protocol* that Section 15 leaves open:
   the fixed 96-subject probe, the four measurement surfaces
   ({eval, train} × {own pairing, production pairing}), the scale-free `posRank`
   statistic, the per-arm InfoNCE null so that E5 at B=128 is comparable to the B=32
   arms, the BatchNorm snapshot/restore around the train-state probe, and the
   abort/incomplete/not-run reporting rules.

   It also lists four numeric thresholds (top-1, posRank, CL, subject rank). Those
   are **descriptive only** and are not used to declare a rescue.

That file is left **exactly as frozen** — it is not edited now that arms have started.
Where the two overlap, Section 15 wins.
