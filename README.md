# apd-bench

> **Start here: [`runbook/README.md`](runbook/README.md).** The study now runs with **no API keys**: three people drive Claude, ChatGPT and Gemini through their free web chats, and all code runs in Google Colab with the lab's Drive folder **mounted** (never uploaded). Notebooks are in [`colab/`](colab/), prompts in [`prompts/agent/`](prompts/agent/), the Colab helper is `src/apd_bench/colab.py`.
>
> The API harness described below (`apd-bench run`, "Arm A") is optional and only usable where you have API keys.

Benchmark harness for the study *Frontier Multimodal Models as Assessors of Acute Psychological Distress Risk: Benchmarking Gemini, Claude and GPT on Synthetic versus Behavioral Video Data*.

It takes a manifest of session recordings (two datasets: behavioral and synthetic), builds three input conditions per session, sends each to every model family under zero- and few-shot prompting with repeats, logs every call, and produces the result tables the research questions ask for.

```
manifest.csv ──► prepare ──► cache/<session>/{transcript, feature sheet, frames}
                                   │
                                   ▼
                run ──► runs/runs.jsonl  (one line per model call; resumable)
                                   │
                                   ▼
             analyze ──► results/{main_table, transfer_gap, rank_agreement, reliability, task_attribution}.csv
           reference ──► results/reference_scores_behavioral.csv  (SVM / GBT nested CV)
```

## Install

```bash
pip install -e ".[dev]"                 # core + tests (runs with the mock model, no keys needed)
pip install -e ".[anthropic,openai,gemini]"   # provider SDKs you have keys for
pip install -e ".[features]"            # openSMILE, librosa, Whisper (needs ffmpeg on PATH)
```
OpenFace is a separate binary; run `FeatureExtraction -f video.mp4 -aus -pose -gaze` and point `openface_csv` in the manifest at its output.

Keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.

## Smoke test (no data, no keys)

```bash
python examples/make_toy_manifest.py examples/toy
apd-bench run examples/toy/manifest.csv --cache cache --out runs --models mock --repeats 2 --skip-media
apd-bench analyze runs/runs.jsonl --out results
pytest
```

## Real run

1. Fill `configs/models.yaml`: pin exact model IDs (the Gemini and GPT IDs are placeholders marked VERIFY), temperature 0, repeats 5.
2. Write `manifest.csv` (columns documented in `src/apd_bench/manifest.py`). Mark 10–20% of each dataset `split=calibration`; only those sessions are ever used as few-shot examples.
3. `apd-bench prepare manifest.csv` — transcribes, builds feature sheets and frames. Labels are never read here.
4. Freeze: commit the manifest and `prompts/`. Their hashes are stamped on every log row.
5. `apd-bench run manifest.csv --models gemini,claude,gpt` — safe to interrupt and rerun.
6. `apd-bench analyze runs/runs.jsonl` and `apd-bench reference manifest.csv`.

## Layout

| Path | What |
| --- | --- |
| `configs/models.yaml` | model families, supported conditions, repeats, frame sampling |
| `prompts/` | system prompt + zero/few-shot templates (JSON output schema) |
| `src/apd_bench/manifest.py` | session schema, manifest hashing |
| `src/apd_bench/preprocess/` | OpenFace AU summaries, eGeMAPS, Whisper, latency, frame sampling, feature sheet |
| `src/apd_bench/models/` | one `Request → Response` interface; Anthropic, OpenAI, Gemini, mock backends |
| `src/apd_bench/runner.py` | the grid, few-shot assembly from calibration split, JSONL logging |
| `src/apd_bench/analysis/metrics.py` | DeLong AUROC CI, paired bootstrap, sens@90% spec, Brier/ECE, κ, Kendall τ |
| `src/apd_bench/analysis/report.py` | result tables per RQ |
| `src/apd_bench/analysis/reference.py` | classical SVM/GBT reference, nested CV |

## Input-parity note

Gemini accepts video natively; Claude and GPT take images. In `raw_media`, Gemini gets the session video, Claude and GPT get evenly sampled frames plus the transcript. The `feature_sheet` and `transcript` conditions are identical text for all three models and are the fair comparison. `raw_media_kind` in the config controls this per model.

## What is deliberately not here yet

- Rationale coding against SHAP importance (RQ3c) — needs human coders; the model `evidence` field is logged for them.
- Prompt-paraphrase robustness — add paraphrased templates under `prompts/` and a `prompt_variant` loop in the runner.
- Mixed-effects model (model × dataset interaction) — do in R/statsmodels on `runs.jsonl`.
