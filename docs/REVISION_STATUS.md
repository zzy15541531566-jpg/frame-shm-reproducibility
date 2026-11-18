# November 2026 revision analyses

## Completed and verified

- The public repository contains code, configuration, data and archived full results.
- The code and associated software documentation are licensed under MIT, copyright 2026 Zhenyu Zhao. Data/result assets have no separately assigned reuse license.
- The full-feature MLP was refitted on all ten original stratified splits. Split accuracy, macro-F1, stopping epoch, all per-class F1 scores, and the pooled confusion matrix match the archive. The new environment is explicitly recorded.
- Location accuracy at nominal 10% loss is 89.1667%, with a descriptive 95% split-level t interval of 86.2657–92.0676%. At nominal 20%/30%, all ten splits give 100%; this is not certainty for new structures.
- Story-specific severity intervals appear in `revision_results/factorized_summary.csv`. They describe reused simulation splits, not engineering uncertainty.
- Modal MLP minus full random forest has mean paired accuracy difference 0.6410 percentage points, corrected interval −0.6694 to 1.9515 points and two-sided p=0.29718. No superiority or equivalence is established.
- Actual damage ratios are logged without adding random draws. Regeneration matches all original samples. Nominal 10%/20%/30% classes realize ranges 6.5262–13.2924%, 16.4663–24.1471%, and 26.4978–33.6000%.
- First-frequency means for nominal 10% damage fall 0.2853–2.0045% relative to the intact mean across stories, with overlapping observed ranges. This is descriptive evidence, not a causal test.
- Figure 4 is regenerated from the archived summary tables; figure 5 uses readable class IDs corresponding to Table I. Exact confusion counts remain in the archived CSV.

## Corrections and provenance

The full random-forest summary is 88.3333% ± 2.6482% (sample SD), replacing an older manuscript transcription of 88.30% ± 2.52%. Logistic accuracy is 86.9551% ± 2.1688%; its macro-F1 is 86.9142% ± 2.2232%. The original data and reference-result files remain byte-identical.

`docs/original_archive_sha256.json` preserves the pre-revision file manifest. `docs/archive_sha256.json` is refreshed for the current source files, including the generator's optional audit-record interface. The simulation algorithm and random-number sequence are unchanged.

No external IASC–ASCE, laboratory or multi-story-damage validation was performed. The manuscript remains subject to author review and target-editor pagination checks. This repository does not contain the manuscript itself.
