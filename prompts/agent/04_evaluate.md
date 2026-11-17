# PHASE 4 — HELD-OUT PREDICTIONS AND ROBUSTNESS

You may now process `data/test/`. You still have no test labels and must not try to infer them.

1. **Freeze.** Train the final model on all training sessions with the settings chosen in Phase 3. Do not tune anything from here on. Record the frozen configuration in decisions.md.
2. **Score the test set.** Write `predictions.csv` (session_id, risk_score 0–100, risk_label) and `channels.csv` (session_id, vision_score, language_score, acoustic_score, fused_score, affect_labels). A channel you did not build is left empty, not zero. A session you could not process gets risk_score 50, risk_label control, and a row in `failures.csv` (session_id, stage, reason). Run `apd.check_contract()` and confirm prediction coverage is complete.
3. **Robustness (training set, out-of-fold).** Using duration_tertile, luminance_tertile and snr_tertile from audit.csv, report AUROC and sensitivity at 90% specificity within each tertile, with confidence intervals. Write `strata_report.md`: one table per variable and one paragraph on what the pattern means — which condition hurts, and which channel degrades.
4. **Channel ablation (training set, out-of-fold).** AUROC with vision only, language only, acoustics only, and with each one removed from the fusion. Add the table to strata_report.md.
5. **Calibration.** Reliability table (deciles of predicted risk vs observed rate) on out-of-fold predictions; say whether risk_score can be read as a probability.

Everything in this phase must be produced by `run.sh`. Finish with the end-of-phase cells.
