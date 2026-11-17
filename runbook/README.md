# Runbook — free-chat protocol (no API keys)

Three people each drive one model through its **free web chat**; all code runs in **Google Colab** with **Google Drive mounted** (videos are never uploaded anywhere). A coordinator prepares the data once and scores at the end.

| Who | Model / chat | Runbook | Colab notebook | Drive output folder |
| --- | --- | --- | --- | --- |
| Coordinator | — | [COORDINATOR.md](COORDINATOR.md) | `colab/00_coordinator_setup.ipynb`, `colab/02_coordinator_score.ipynb` | `apd_benchmark/` (shared, read-only), `apd_private/` (never shared) |
| Person 1 | Claude — claude.ai | [OPERATOR_1_CLAUDE.md](OPERATOR_1_CLAUDE.md) | `colab/01_operator_workspace.ipynb` with `MODEL = "claude"` | `apd_outputs_claude/` |
| Person 2 | ChatGPT — chatgpt.com | [OPERATOR_2_GPT.md](OPERATOR_2_GPT.md) | same, `MODEL = "gpt"` | `apd_outputs_gpt/` |
| Person 3 | Gemini — gemini.google.com | [OPERATOR_3_GEMINI.md](OPERATOR_3_GEMINI.md) | same, `MODEL = "gemini"` | `apd_outputs_gemini/` |

## How one cycle works

```
 chat (model writes)            you (copy / paste)             Colab (runs, Drive mounted)
 ───────────────────            ──────────────────             ───────────────────────────
 CELL 3: %%writefile ...   ──►  paste into an empty cell  ──►  file saved to outputs/<model>/ on Drive
 CELL 4: apd.run("python…") ─►  paste, run                ──►  script reads data/ (read-only Drive)
                           ◄──  copy COPY FROM HERE…END   ◄──  short summary printed; full log on Drive
 next cells …
```

The model never sees video, audio, frames or transcripts — only what its own code prints. All state lives on Drive (`STATE.md`, `decisions.md`, caches), so a usage limit, a closed laptop or a Colab reset loses nothing: run setup again and paste `apd.resume_packet()` into a new chat.

## Order and timing

1. Coordinator: setup notebook (half a day, mostly waiting for copies and the audit).
2. Three operators in parallel: phases 1–6. Plan on 3–6 sittings each over about a week; long feature extraction runs while you do other things.
3. Coordinator: scoring notebook, blind rating, cross-reproduction.

## Why this is fair and leak-free

- Same charter, same protocol, same data, same compute (free Colab), same operator rules for all three models.
- Test labels exist only in the coordinator's private folder; operators' notebooks refuse to start if they can see that folder or the original labelled `video` folder.
- Operators may only type the fixed messages in their runbook; they never edit code or hint at methods.
- Every chat event (start, limit, downgrade, refusal, new chat) is logged in `operator_notes.md` and reported as a result.
