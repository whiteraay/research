# PHASE 2 — METHODOLOGY PROPOSAL (commit before you build)

Write `outputs/<your_name>/proposal.md` (send it as one `%%writefile` cell). This is your pre-registration: once you start building, every departure from it must be logged in decisions.md with a reason. You may run small feasibility checks first (does a package install on Colab CPU? how long does one session take?) — keep them under 30 minutes in total and log them.

Cover each point. Where you choose between alternatives, name at least two you considered and why you rejected them.

1. **Unit of analysis.** Whole session, fixed-length windows, speaker turns, or a hierarchy? How sessions of very different length are handled so long interviews do not dominate.
2. **Visual channel.** Which face detector / landmark / action-unit / expression / embedding approach, on which frames, at what sampling rate. How you deal with dim lighting, backlighting, several faces, interviewer in frame, and sessions where the face is rarely found. Whether you also use head motion, gaze or posture.
3. **Speech-to-text.** Which recogniser and model size (it must run within budget on free Colab), how you handle mixed languages, how you separate participant from interviewer, what you do when transcription quality is low.
4. **Language channel.** What you extract from the transcript (lexical features, sentiment/emotion classifiers, multilingual text-encoder embeddings, …) and how a test session's transcript influences only that session's prediction.
5. **Acoustic channel.** Which paralinguistic features or audio embeddings, and how you separate the person's voice from microphone and room effects.
6. **Fusion and classifier.** Early, late or learned fusion; classifier family; how hyperparameters are chosen without touching the test set. With a few hundred sessions, justify the model's capacity.
7. **Validation.** Grouped K-fold by participant_id; exact K; repeated or not; how you report variance across folds.
8. **Confound control.** How you show the system is not reading duration, lighting or microphone placement: beat the confound floor from Phase 1, and at least one more test (e.g. adjust for audit variables, or evaluate within tertiles).
9. **Affect analysis plan.** Which affective constructs (low mood/sadness, flat affect, anxiety/tension, withdrawal, at least one mixed state such as masked distress), measured how in face, voice and text, and how you will test where vision and language agree or disagree.
10. **Budget.** Expected CPU/GPU hours per step against the 24 CPU-h / 8 GPU-h limit, and how each long step resumes after a Colab disconnect.
11. **Risks.** The three most likely ways this plan fails and what you will do in each case.

End with a one-paragraph summary as you would write it in an abstract. Under 1,500 words. Then the end-of-phase cells. Do not start building until proposal.md exists.
