# Agent prompts (free-chat protocol)

| File | Sent when | How the operator gets it |
| --- | --- | --- |
| `00_charter.md` + `00b_executor_preamble.md` | first message of the very first chat | `apd.prompt(0)` (both together, model name filled in) |
| `01_audit.md` … `06_report.md` | one per phase, after the previous phase is complete | `apd.prompt(1)` … `apd.prompt(6)` |
| `00c_resume.md` | first message of any new chat after a limit, reset, new day or new phase | `apd.resume_packet()` (charter + protocol + STATE.md + decisions.md + file listing) |

`<your_name>` is replaced automatically with `claude`, `gpt` or `gemini`. Never edit a prompt for one model only; all three must receive identical text.

Full procedure: [`../../runbook/README.md`](../../runbook/README.md).
