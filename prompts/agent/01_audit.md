# PHASE 1 — DATA AUDIT

Before proposing any method, understand the data — through code, since you cannot look at it. Deliverable: `outputs/<your_name>/audit_notes.md` (plus the small scripts you used, under `src/audit/`). Do not extract model features or train anything in this phase.

1. **Inventory.** Write a script that prints: sessions per cohort (train) and in total (test); participants per cohort; sessions per participant; file formats, resolution, frame rate; duration distribution (min, quartiles, max) for train and test.
2. **Recording conditions.** Read `data/audit.csv`. For duration_s, mean_luminance, face_detect_rate and audio_snr_db print the distribution, and in the training set test whether each differs between cohorts (effect size + a simple test). Also fit a cross-validated (grouped by participant) logistic regression on audit.csv columns alone and print its AUROC: this is the **confound floor** every later model must beat. If a recording variable predicts the cohort on its own, say so plainly — it is a confound to handle, not a feature to exploit.
3. **What is in the recordings.** Since you cannot watch them, measure them on a random sample of at least 10 training sessions per cohort (sample with a fixed seed; do not choose by label-related criteria):
   - faces per frame (on ~20 sampled frames per video): how often 0, 1, 2+ faces; size of the largest face relative to the frame;
   - voice activity: fraction of time with speech; number of speech segments; any evidence of two speakers (e.g. pitch clusters or a quick diarisation);
   - language: run a speech recogniser on a 30-second clip per session and print only the detected language and its probability, and the word count;
   - anything likely to break your later pipeline (decode errors, variable frame rate, no audio track, rotated video).
   Print only aggregate tables, never transcript text.
4. **Quality problems and treatment.** List each problem found and whether you will exclude, repair, or model it as a covariate. Exclusion must depend on quality alone, never on the label; state the exact rule.
5. **Strata.** Confirm you will use duration_tertile, luminance_tertile and snr_tertile from audit.csv for all robustness analyses, and print the train/test counts per tertile.

`audit_notes.md`: under 800 words plus tables, written from the numbers you printed. Finish with the end-of-phase cells from the protocol.
