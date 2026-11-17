# PHASE 3 — BUILD AND TRAIN

Implement proposal.md under `outputs/<your_name>/src/`, driven by `outputs/<your_name>/run.sh`. Use only `data/train/` in this phase; you may open `data/test/` files only to confirm they decode.

Requirements:
- `run.sh` installs `requirements.txt`, then runs every step in order. It accepts `--reuse-cache` (skip feature extraction already cached) and `--limit N` (first N sessions per split, for quick checks).
- Feature extraction caches per session under `outputs/<your_name>/cache/<step>/<session_id>.*` and skips cached sessions, so a Colab reset costs nothing. Print progress, not per-frame output.
- Name every pre-trained model with its exact version in `requirements.txt` or `MODELS.md`.
- Grouped cross-validation by participant_id exactly as proposed. Write `train_cv_predictions.csv` (session_id, fold, risk_score, risk_label) and per-fold AUROC, sensitivity at 90% specificity and Brier score to `cv_results.json`, with the confound floor computed on the same folds alongside.
- Append each step's wall-clock and GPU time to `budget.csv`.
- Keep decisions.md current: dated entries for each change, why, and its cross-validated effect.

Work in this order: (a) run each extractor with `--limit 3` and check its printed summary; (b) full extraction, possibly across several Colab sessions; (c) model training and cross-validation.

When cross-validation is done, write `phase3_summary.md`: final cross-validated numbers with variance, the three most important features or signals, and whether you beat the confound floor. If you do not, say so, decide what you will change, log it, and change it before Phase 4.

Finish with the end-of-phase cells. Do not score the test set yet.
