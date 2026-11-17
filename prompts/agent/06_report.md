# PHASE 6 — FINAL REPORT AND REPRODUCIBILITY

1. **methods.md** — the Methods section of a paper, 1,000–1,800 words: data and splits; preprocessing and QC with exclusion counts; each channel's extraction with named models and versions; fusion and classifier; validation design; confound handling; affect-analysis method; compute used (from budget.csv). Past tense, no promotional language, every number traceable to a file your code wrote.
2. **decisions.md** — turn the running log into a dated table: decision · alternatives considered · reason · cross-validated effect. Add the decisions you would make differently with more time or data.
3. **README.md** — how to reproduce everything with `run.sh` on a fresh Colab runtime, expected runtime with and without `--reuse-cache`, and a one-paragraph plain-language summary of what the system does and does not do. Then answer, each in under 150 words, as honestly as you can:
   1. What is the single strongest piece of evidence that your system detects distress rather than a recording artefact?
   2. Which recording condition hurts it most, and would you trust it on a short, dim, noisy interview?
   3. If a clinician asked what your system actually looks at, what would you tell them?
4. **Reproducibility check.** The operator will reset the Colab runtime ("Disconnect and delete runtime"), re-run the setup cell, and run `apd.run("bash outputs/<your_name>/run.sh --reuse-cache")`. Give that cell, then a cell that compares the regenerated predictions.csv with the previous one (save a copy first as `predictions_submitted.csv`) and prints the maximum absolute difference in risk_score. Explain any difference.
5. Final cells: `apd.check_contract()`, update STATE.md to `phase: done`, and `apd.ls()`. Say "PROJECT COMPLETE" and report total compute used.
