# Revision status

This preparation package restores the directory layout expected by the original code; the numerical algorithms and source modules are unchanged.

## Checked

- Original data and configuration are included together.
- Generation seed is 20260708; actual damage clipping bounds are 0.04 and 0.42.
- A previous in-memory regeneration matches all 1,560 dataset rows within numerical precision.
- The archived full random-forest summary is 88.3333% ± 2.6482% (sample SD), which differs from an older manuscript value of 88.30% ± 2.52%.

## To complete before a paper-specific release

1. Recover per-split predictions and verify their aggregate confusion matrix against the archive.
2. Add factorized accuracy intervals and a paired model comparison with appropriate dependence qualifications.
3. Record realized damage ranges without changing the generator's random-number sequence.
4. Update the paper and enlarge the relevant figure labels.
5. Select and add code/data licenses, publish the repository, and record an exact version in the paper's availability statement.

No external IASC–ASCE or laboratory validation has been completed.
