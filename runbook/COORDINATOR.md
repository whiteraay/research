# Coordinator runbook

You prepare the data **once**, keep the test labels, and score at the end. You never run a model chat yourself while you hold the labels.

## A. One-time preparation (≈ half a day, mostly waiting)

1. **Make the lab folder visible to Colab.** Drive → *Shared with me* → `video` → ⋮ → *Organize* → *Add shortcut* → *My Drive*. (Folder id `16U1RUVTP25s-7TjCsIP89HIHGcOyHtrg`. Colab only sees My Drive; the shortcut is how a shared folder gets mounted.)
2. **Get the notebook.** Open `colab/00_coordinator_setup.ipynb` from GitHub in Colab: colab.research.google.com → *File* → *Open notebook* → *GitHub* → `whiteraay/research` (tick *Include private repos* and sign in), or upload the `.ipynb` once from your computer.
3. **Code access in Colab.** Either add a Colab secret `GITHUB_TOKEN` (🔑 on the left; a GitHub fine-grained token with *read* access to `whiteraay/research`), or unzip the repo into `MyDrive/apd_benchmark/code/research` yourself before cell 1.
4. **Run cells 1–2 (dry run).** Read the participant table. Each real participant must get one id, and no id may appear in both cohorts. If file names are like `12_interview.mp4`, the default works; if they are like `interview_12.mp4`, set `PARTICIPANT_REGEX = r"_(\d+)$"`; if each participant has a sub-folder, it is handled automatically. Ask if unsure — a wrong grouping leaks people across train and test.
5. **Run cell 3 (split).** Copies inside Drive; may take an hour or more for many GB. If Colab disconnects, re-run cells 1 and 3; finished files are skipped.
6. **Run cell 4 (audit)** — resumable the same way. Then cell 5: write down any recording variable with AUROC above ~0.65 on its own; that is a confound and goes into the paper.
7. **Run cell 6** (code copy for operators + preflight). It must print `PREFLIGHT: OK`.
8. **Lock down and share** (cell 7 instructions):
   - Share **`apd_benchmark`** with the three operators as **Viewer**.
   - Never share **`apd_private`**.
   - If operators will use the **same Google account** as you: run cell 8 to download `test_labels.csv` and `split_manifest.csv`, keep them on your own computer, delete `apd_private` from Drive and empty the trash.
   - Remove the **`video`** shortcut from My Drive (the lab folder stays shared; only your shortcut goes).
9. Send each operator their runbook file and the name of their model.

## B. During the runs

- Do not answer methodological questions from operators; the answer is always "the model decides".
- Collect each `proposal.md` as soon as Phase 2 is done (operators share their `apd_outputs_<model>` folder with you as Viewer at the start; add a shortcut to My Drive). Proposals are the pre-registration.
- Check `operator_notes.md` once a day for downgrades or long gaps.

## C. Scoring (after all three say PROJECT COMPLETE)

1. Open `colab/02_coordinator_score.ipynb`, run cell 1.
2. Cell 2 finds `apd_private/test_labels.csv` or asks you to upload it (a tiny label file).
3. Cells 3–6: collect, contract check, score, tables. Results in `results/arm_b/` (`summary.csv`, `strata.csv`, `pairwise.csv`, `confound_floor.json`).
4. **Blind rating:** copy the three output folders to `A/`, `B/`, `C/` in a random order (write the key down privately), redact model names, and have two raters score the qualitative rubric rows.
5. **Cross-reproduction:** operator 1 runs operator 2's `run.sh --reuse-cache`, 2 runs 3's, 3 runs 1's, each in their own Colab, and reports the max difference in `risk_score`.
6. Commit code changes (not data or outputs) to GitHub:
   ```bash
   git add -A && git commit -m "Arm B scoring" && git push
   ```
