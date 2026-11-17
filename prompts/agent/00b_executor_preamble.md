# EXECUTION PROTOCOL — how you and the operator work together

The operator has a Colab notebook open with Google Drive mounted and the workspace set up (`/content/research`, with `data/` and `outputs/<your_name>/` ready). The operator will run what you give them exactly as written and paste back what it printed. The operator will not edit your code, fix your bugs, choose between options for you, or describe the data to you. If a decision is needed, make it yourself and log it in decisions.md.

## How to give code
Every code block you send must be ONE complete Colab cell, labelled `CELL 1`, `CELL 2`, … in the line above it. Use only these three kinds:

1. A file, written to disk — first line `%%writefile <path>`, then the complete file contents:
   ```
   %%writefile outputs/<your_name>/src/extract_audio.py
   ...full file...
   ```
   Always send the whole file, never a diff or "change line 40". To change a file, resend it in full.
   `%%writefile` cannot create folders. `src/`, `src/audit/`, `cache/`, `figures/` and `logs/` already exist; for any other folder, first send a cell `apd.run("mkdir -p outputs/<your_name>/<folder>")`.

2. A command, run with the project helper (this saves the full log to Drive and prints a short paste-ready summary):
   ```
   apd.run("python outputs/<your_name>/src/extract_audio.py --limit 5")
   ```
   Use `apd.run("...")` for every script and shell command, including `pip install -r outputs/<your_name>/requirements.txt` and `bash outputs/<your_name>/run.sh`.

3. A helper call: `apd.ls()` (list your outputs), `apd.check_contract()` (check required files and columns).

Do not send interactive Python cells that define state in the notebook; the runtime can reset at any time. Put logic in files, run files.

## What the operator pastes back
The summary between `COPY FROM HERE` and `END OF COPY`: the command, exit code, run time, the first 40 and last 60 lines of output (long lines cut). The full log is in `outputs/<your_name>/logs/` if you need a specific part — ask for it with a command such as `apd.run("grep -n Error outputs/<your_name>/logs/cmd_0012.log | head -50")`.

Design your scripts to print compact summaries: counts, shapes, means, metrics, a few rows of a table. Write details to files and print where they went.

## What must never reach this chat
Your code must never print: raw frames or images, audio, transcript text (beyond word counts, detected language, and at most one 12-word sample per session), names, or anything identifying. You will not be shown figures; if you make plots, save them and describe them from the numbers you printed.

## Long jobs
Free Colab disconnects after inactivity and caps session length. For anything that takes more than about 20 minutes: process sessions one at a time, save each result to `outputs/<your_name>/cache/` as soon as it is computed, skip sessions already cached, and print progress every N sessions. First run on `--limit 3` to check it works, then the full run.

## GPU
CPU is the default. If a step truly needs a GPU, say so in one line; the operator will try to switch the runtime to T4 and tell you whether it worked. Always keep a CPU path.

## State and interruptions
The chat has usage limits and may be replaced by a new conversation at any time. Keep `outputs/<your_name>/STATE.md` current after every meaningful step, in this format:

```
phase: <1-6>
done: <bullet list of completed steps with output files>
in progress: <what is half-done and how to resume it>
next command: <the exact next cell to run>
open issues: <anything unresolved>
```

Keep `outputs/<your_name>/decisions.md` as a dated log of every methodological decision. A new conversation will be given only the charter, this protocol, STATE.md, decisions.md and a file listing, so those two files must be enough to continue.

## End of each phase
End every phase with a cell that updates STATE.md and then a cell `apd.ls()`, and say clearly "PHASE N COMPLETE".

Now acknowledge the charter and this protocol together in under 200 words: what you are asked to produce, what you are free to decide, and the constraints you will design around. Then wait for Phase 1.
