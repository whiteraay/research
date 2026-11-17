# PHASE 5 — WHAT THE SYSTEM SEES: AFFECT IN FACE, VOICE AND WORDS

This phase answers the study's central secondary question: not only whether the cohorts can be separated, but which affective signal does the separating, and whether the visual and language channels are reading the same thing. Deliverables: `affect_analysis.md`, `affect_features.csv`, and the code under `src/`. Use training sessions only (labels known); never use or guess test labels.

1. **Define the constructs.** Operationalise, in each channel where it makes sense: low mood / sadness; flat or blunted affect; anxiety or tension; withdrawal or disengagement; and at least one mixed or complex state you can justify (e.g. masked distress: neutral or positive words with flat face and voice). State exactly how each is measured — action units or an expression model for the face; pitch range, energy, pauses and speech rate for voice; an emotion or sentiment model, pronoun use and lexical valence for text (multilingual where needed).
2. **Measure them.** One row per training session in `affect_features.csv`: session_id, cohort, each construct × channel (face_sadness, voice_flatness, text_sadness, …), plus duration_s, mean_luminance, audio_snr_db.
3. **Which constructs separate the cohorts?** Effect size (Cohen's d or AUROC) per construct per channel, with confidence intervals, adjusted for duration_s and audio_snr_db. Print a ranked table.
4. **Where do vision and language agree?** Within session, correlate face, voice and text scores for the same construct. Define a disagreement score (e.g. face/voice say flat or sad while text says neutral or positive, and the reverse). Test whether disagreement predicts the cohort over and above the channels themselves (nested model comparison, grouped CV). This vision × language intersection is the result the study is after; treat it carefully.
5. **What does the fused model use?** Relate the Phase 4 model's feature importances or attributions to the construct scores, and say in plain words what it has learned to respond to.
6. **Confound check.** Show whether the top constructs still separate the cohorts within each luminance and SNR tertile.
7. **Limits.** Constructs you could not measure reliably, and why (detector failure in dim light, language coverage, short sessions).

You may save figures under `outputs/<your_name>/figures/`; you will not see them, so describe them from printed numbers. `affect_analysis.md` under 2,000 words. Finish with the end-of-phase cells.
