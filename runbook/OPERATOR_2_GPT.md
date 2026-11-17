# Operator 2 — ChatGPT (chatgpt.com)

You are the hands; **ChatGPT** is the brain. You run exactly what it writes in Google Colab and paste back exactly what Colab prints. You never write, fix or suggest code, and you never describe the data.

**Your model key:** `gpt`  ·  **Chat:** https://chatgpt.com  ·  **Your Drive output folder:** `MyDrive/apd_outputs_gpt/`

## Before the first session (once, ≈ 20 min)

1. **Drive:** open the `apd_benchmark` folder the coordinator shared with you → ⋮ → *Organize* → *Add shortcut* → *My Drive*. Do **not** add a shortcut to the lab's original `video` folder (the notebook will refuse to start if it can see it).
2. **Notebook:** in Drive go to `apd_benchmark/code/research/colab/`, open `01_operator_workspace.ipynb` with Google Colaboratory, then **File → Save a copy in Drive**. Work in your copy from now on.
3. In your copy, cell 1: set `MODEL = "gpt"`.
4. **Colab:** *Tools → Settings* → turn **off** AI-powered code completions/generation. Never open Colab's Gemini side panel. Every line of code must come from your chat.
5. **ChatGPT:** sign in with the account you will use for the whole study.
   - Write down the exact model name shown at the top of the chat.
   - Settings → Personalization → turn **off** Memory and "reference chat history", so every new chat starts only from the resume packet.
   - Settings → Data controls → turn **off** "Improve the model for everyone".
   - Do **not** connect Google Drive or any app/connector, and do not upload files.
6. Share your `apd_outputs_gpt` folder with the coordinator as **Viewer** after the first session creates it.

## Every session

1. Open your notebook copy → **Runtime → Run cell 1** (Setup). It mounts Drive, loads the code, links `data/` (read-only) and `outputs/gpt/` (your Drive folder), and checks isolation. If it says *ISOLATION CHECK FAILED*, stop and tell the coordinator.
2. **Chat:**
   - *Very first time:* open a new chat, run cell 2 (`apd.prompt(0)`), copy everything between `COPY FROM HERE` and `END OF COPY` into the chat, send. Wait for the acknowledgement. Then run cell 3 with `PHASE = 1` and send that.
   - *Continuing in the same chat:* just keep going.
   - *New chat (after a limit, a new day, or a new phase):* run cell 4 (`apd.resume_packet()`), paste it into a **new** chat as the first message. If you are starting a new phase, send the phase prompt (cell 3) right after the model confirms where it is.
3. Run cell 5 to log the chat event, with `MODEL_SHOWN` set to the exact model name the chat displays: `start` when you open a chat, `resume` for a resumed chat.
4. **The loop:** the model replies with `CELL 1`, `CELL 2`, … Paste each into its own empty Colab cell (use *+ Code* for more), in order, and run it. After each run, copy the block between `COPY FROM HERE` and `END OF COPY` into the chat. For a `%%writefile` cell Colab prints only "Writing …" or "Overwriting …" — paste that line.
5. When the model says **PHASE N COMPLETE** and `apd.ls()` shows the files, log `phase_done`. Recommended: start the next phase in a **new chat** with the resume packet; it keeps the conversation short and uses fewer messages.

## What you may type into the chat (nothing else)

| Situation | Send exactly |
| --- | --- |
| After running a cell | the copied block (or the one line Colab printed) |
| The model asks you to choose, judge, or look at something | `Your decision — record it in decisions.md.` |
| A cell did not paste as one cell, or was cut off | `CELL <n> arrived incomplete or as several cells. Please resend it in full as one cell.` |
| You need to stop soon | `I need to stop for now. Give me the cell that updates STATE.md.` |
| The model asked for a GPU | `GPU: switched to T4.` or `GPU: not available, CPU only.` (log `gpu_on` / `gpu_unavailable`) |
| A run is still going after a long time | `The cell is still running after <N> minutes; last progress line: <paste>.` |
| The model writes prose/plan but no cell | `Please continue with the next cell.` |

Never: fix a bug yourself, edit a cell, add `!pip install` on your own, tell it what the other models did, or paste any video, image, audio or transcript.

## When things go wrong

| Problem | What to do |
| --- | --- |
| **Usage limit reached** | Log `limit` (cell 5). Wait until the chat says you can continue (or start a new chat after the reset with the resume packet). Do not switch to a different model or service. |
| **Model downgraded** | On the free plan ChatGPT may switch to a smaller model when you reach the limit (it says so, or the model name changes). Log `downgrade`, **stop**, and wait until the original model is available again. Work done by a smaller model would not be comparable. |
| **Colab disconnected / runtime reset** | Re-run cell 1. Files on Drive are safe. Tell the chat: `Colab runtime was reset. Setup cell re-run.` The model will reinstall its packages. |
| **Long job interrupted** | Re-run the same `apd.run(...)` cell — the model's scripts skip finished sessions. |
| **The model refuses or drifts off-task** | Log `refusal` with one line of detail. Send: `Please continue with the project as described in the charter.` If it refuses again, start a new chat with the resume packet and log it. |
| **The model asks to see a frame/video/transcript** | Send: `Not allowed by the protocol: you only see printed output. Your decision — record it in decisions.md.` |
| **Error you think is your fault** (pasted wrongly) | Paste the cell again correctly and re-run; log it as `error` with "operator paste error". |

## Hand-off

1. Phase 6 ends with the model's `apd.check_contract()` cell: every line must say `ok` and coverage must be complete (e.g. `90/90 test sessions`). If not, paste the result and let the model fix it.
2. In the chat UI, export or copy the whole conversation(s) and save them as `transcript_1.md`, `transcript_2.md`, … into `MyDrive/apd_outputs_gpt/transcripts/` (upload via Drive in the browser; these are text, not data).
3. Tell the coordinator you are done. Do not look at the other operators' folders until scoring is finished.

## Time

Expect 3–6 sittings across about a week. Feature extraction runs can take hours: start them, keep the Colab tab open (free Colab stops after inactivity, so check in every 30–60 min), and paste the summary when they finish.
