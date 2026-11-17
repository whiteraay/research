"""Colab helpers for the no-API ("free chat") protocol.

Everyone runs code in Google Colab with Google Drive MOUNTED (never uploaded).
The chat model (Claude / ChatGPT / Gemini, free tier, in its normal web chat)
writes code; the operator pastes it into Colab, runs it, and pastes the printed
result back. These helpers make that loop safe and repeatable:

    from apd_bench import colab as apd

    # coordinator (once)
    apd.mount(); apd.coordinator_dry_run(); apd.coordinator_split(); apd.coordinator_audit()

    # operator (every Colab session)
    apd.setup_workspace("claude")          # or "gpt" / "gemini"
    apd.prompt(0)                          # print a prompt, ready to paste into the chat
    apd.run("python outputs/claude/src/extract.py")   # run model code, print a paste-ready summary
    apd.resume_packet()                    # after a usage limit / new chat
    apd.log_chat("limit")                  # record chat events in operator_notes.md
    apd.check_contract()                   # before hand-off

Drive layout
    MyDrive/video/{Experimental,Control}      original lab folder (shortcut)   coordinator only
    MyDrive/apd_benchmark/data/               split + audit, shared READ-ONLY with operators
    MyDrive/apd_benchmark/code/               copy of this repo for operators without GitHub access
    MyDrive/apd_private/                      test labels + split manifest     coordinator only
    MyDrive/apd_outputs_<model>/              one operator's outputs (persists across Colab resets)
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- locations
SOURCE_FOLDER_ID = "16U1RUVTP25s-7TjCsIP89HIHGcOyHtrg"   # lab "video" folder (Experimental/, Control/)
SOURCE_FOLDER_NAME = "video"
MODELS = ("claude", "gpt", "gemini")

DRIVE = Path(os.environ.get("APD_DRIVE_ROOT", "/content/drive"))
MYDRIVE = DRIVE / "MyDrive"
BENCH = MYDRIVE / "apd_benchmark"
PRIVATE = MYDRIVE / "apd_private"
REPO = Path(os.environ.get("APD_REPO", Path(__file__).resolve().parents[2]))
PROMPTS = REPO / "prompts" / "agent"

_MODEL: Optional[str] = None

PASTE_START = "=" * 20 + " COPY FROM HERE INTO THE CHAT " + "=" * 20
PASTE_END = "=" * 20 + " END OF COPY " + "=" * 20


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- mounting
def mount(force_remount: bool = False) -> Path:
    """Mount Google Drive at /content/drive. Outside Colab, uses APD_DRIVE_ROOT as-is (for testing)."""
    if _in_colab():
        from google.colab import drive
        drive.mount(str(DRIVE), force_remount=force_remount)
    elif not DRIVE.exists():
        raise FileNotFoundError(f"Not in Colab and {DRIVE} does not exist; set APD_DRIVE_ROOT for local tests.")
    return DRIVE


def _shortcut_targets(*parts: str) -> list[Path]:
    root = DRIVE / ".shortcut-targets-by-id"
    if not root.exists():
        return []
    return sorted(root.glob("*/" + "/".join(parts)))


def find_source_folder() -> Optional[Path]:
    """The ORIGINAL labelled folder (has Experimental/ and Control/). Coordinator only."""
    candidates = [MYDRIVE / SOURCE_FOLDER_NAME]
    candidates += sorted((DRIVE / ".shortcut-targets-by-id" / SOURCE_FOLDER_ID).glob("*")) \
        if (DRIVE / ".shortcut-targets-by-id" / SOURCE_FOLDER_ID).exists() else []
    candidates += _shortcut_targets(SOURCE_FOLDER_NAME)
    for c in candidates:
        if (c / "Experimental").is_dir() and (c / "Control").is_dir():
            return c
    return None


def find_benchmark_data() -> Optional[Path]:
    """The shared, already-split data folder (train/, test/, audit.csv)."""
    for c in [BENCH / "data", *_shortcut_targets("apd_benchmark", "data")]:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    return None


# ---------------------------------------------------------------- isolation (no test-label leakage)
LEAK_NAMES = ("test_labels", "split_manifest")


def isolation_report(data_dir: Path) -> list[str]:
    """Problems that would let test labels reach an operator's runtime. Empty list = clean."""
    problems = []
    src = find_source_folder()
    if src is not None:
        problems.append(
            f"The original labelled folder is visible at {src}. Its Experimental/Control folders reveal "
            f"which test videos are which. Remove the '{SOURCE_FOLDER_NAME}' shortcut from My Drive "
            f"(right-click > Remove; the files stay shared) and remount."
        )
    if PRIVATE.exists():
        problems.append(f"{PRIVATE} is visible from this account. Move it to the coordinator's own account or download it and delete it from Drive.")
    for p in data_dir.rglob("*"):
        if any(n in p.name for n in LEAK_NAMES):
            problems.append(f"Private file inside the shared data folder: {p}")
    for p in MYDRIVE.glob("*"):
        if any(n in p.name for n in LEAK_NAMES):
            problems.append(f"Private file at the top of My Drive: {p}")
    return problems


# ---------------------------------------------------------------- operator workspace
def _out_dir(model: str) -> Path:
    return MYDRIVE / f"apd_outputs_{model}"


def _link(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a link; move it away first.")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def setup_workspace(model: str, allow_unsafe: bool = False) -> dict:
    """Run at the start of EVERY Colab session (Colab forgets everything outside Drive)."""
    global _MODEL
    model = model.lower().strip()
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    mount()
    data = find_benchmark_data()
    if data is None:
        raise FileNotFoundError(
            "Shared benchmark data not found. In Drive, open the 'apd_benchmark' folder the coordinator shared "
            "with you, choose 'Add shortcut to Drive' > My Drive, then re-run this cell."
        )
    problems = isolation_report(data)
    if problems:
        print("ISOLATION CHECK FAILED:")
        for p in problems:
            print("  -", p)
        if not allow_unsafe:
            raise RuntimeError("Fix the problems above before starting (see runbook). Nothing was changed.")
        print("Continuing because allow_unsafe=True. Record this in operator_notes.md.")

    others = [m for m in MODELS if m != model and _out_dir(m).exists()]
    if others:
        print(f"Note: output folders of {others} are visible from this account (shared lab account?). "
              "That is allowed, but never open them or mention them in the chat until scoring is done.")

    out = _out_dir(model)
    for sub in ("logs", "cache", "src", "src/audit", "figures"):   # %%writefile cannot create folders
        (out / sub).mkdir(parents=True, exist_ok=True)
    Path("/content/tmp").mkdir(parents=True, exist_ok=True) if _in_colab() else None
    _link(REPO / "data", data)
    (REPO / "outputs").mkdir(exist_ok=True)
    _link(REPO / "outputs" / model, out)
    os.chdir(REPO)
    _MODEL = model

    notes = out / "operator_notes.md"
    if not notes.exists():
        notes.write_text(f"# Operator notes — {model}\n\n| time | event | detail |\n| --- | --- | --- |\n")
    note("colab session start", f"isolation {'OK' if not problems else 'UNSAFE (allowed)'}"
         + (f"; other outputs visible: {', '.join(others)}" if others else ""))

    n_train = sum(1 for p in (data / "train").rglob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXT)
    n_test = sum(1 for p in (data / "test").glob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXT)
    info = {"model": model, "repo": str(REPO), "data": str(data), "outputs": str(out),
            "train_videos": n_train, "test_videos": n_test, "audit": (data / "audit.csv").exists(),
            "state": (out / "STATE.md").exists()}
    print(f"Workspace ready for {model}.")
    for k, v in info.items():
        print(f"  {k:13s} {v}")
    if info["state"]:
        print("\nSTATE.md exists: this is a resumed run. In a NEW chat paste apd.resume_packet(); "
              "in the SAME chat just keep going.")
    return info


def _model() -> str:
    if _MODEL is None:
        raise RuntimeError("Call apd.setup_workspace('<model>') first.")
    return _MODEL


def _out() -> Path:
    return REPO / "outputs" / _model()


# ---------------------------------------------------------------- notes
def note(event: str, detail: str = "") -> None:
    p = _out() / "operator_notes.md"
    with p.open("a") as f:
        f.write(f"| {_now()} | {event} | {detail.replace('|', '/')} |\n")


def log_chat(event: str, model_shown: str = "", detail: str = "") -> None:
    """Record chat events. event: start | limit | resume | downgrade | refusal | error | phase_done"""
    note(f"chat {event}", f"{model_shown} {detail}".strip())
    print(f"logged: chat {event} {model_shown} {detail}".strip())


# ---------------------------------------------------------------- prompts
_PROMPT_FILES = {
    0: "00_charter.md", "charter": "00_charter.md",
    "preamble": "00b_executor_preamble.md",
    1: "01_audit.md", 2: "02_proposal.md", 3: "03_build.md",
    4: "04_evaluate.md", 5: "05_affect.md", 6: "06_report.md",
}


def _prompt_text(key) -> str:
    return (PROMPTS / _PROMPT_FILES[key]).read_text().replace("<your_name>", _model())


def prompt(key) -> None:
    """Print a prompt with <your_name> filled in. prompt(0) prints charter + execution protocol together."""
    text = _prompt_text(0) + "\n\n---\n\n" + _prompt_text("preamble") if key in (0, "charter") else _prompt_text(key)
    print(PASTE_START)
    print(text)
    print(PASTE_END)
    note("prompt shown", str(key))


# ---------------------------------------------------------------- running model code
def _clip(line: str, max_chars: int) -> str:
    line = line.rstrip("\n")
    return line if len(line) <= max_chars else line[:max_chars] + f" …[+{len(line) - max_chars} chars]"


def run(cmd: str, head: int = 40, tail: int = 60, max_line_chars: int = 300,
        live: bool = True, timeout: Optional[int] = None) -> int:
    """Run a shell command in the repo root (exactly as the model wrote it), save the full log to Drive,
    and print a short paste-ready summary for the chat. Returns the exit code."""
    out = _out()
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    n = len(list(logs.glob("cmd_*.log"))) + 1
    log_path = logs / f"cmd_{n:04d}.log"
    first: list[str] = []
    last: deque[str] = deque(maxlen=tail)
    total = 0
    t0 = time.time()
    with log_path.open("w") as log:
        log.write(f"$ {cmd}\n")
        proc = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace")
        try:
            for line in proc.stdout:
                log.write(line)
                total += 1
                if live:
                    print(line, end="")
                if len(first) < head:
                    first.append(line)
                else:
                    last.append(line)
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
    wall = time.time() - t0
    with (out / "logs" / "commands.csv").open("a", newline="") as f:
        csv.writer(f).writerow([_now(), cmd, rc, round(wall, 1), log_path.name])

    print("\n" + PASTE_START)
    print(f"$ {cmd}")
    print(f"[exit code {rc}, {wall:.0f}s, {total} lines; full log saved as outputs/{_model()}/logs/{log_path.name}]")
    for line in first:
        print(_clip(line, max_line_chars))
    skipped = total - len(first) - len(last)
    if skipped > 0:
        print(f"…[{skipped} lines omitted]…")
    for line in last:
        print(_clip(line, max_line_chars))
    print(PASTE_END)
    return rc


def ls(sub: str = "", max_lines: int = 150) -> None:
    """Paste-ready file listing of outputs/<model>/ (skips cache/ and logs/ contents)."""
    print(PASTE_START)
    print(_listing(_out() / sub, max_lines))
    print(PASTE_END)


def _listing(root: Path, max_lines: int = 150) -> str:
    lines = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        rel = Path(dirpath).relative_to(root)
        if rel.parts and rel.parts[0] in ("cache", "logs"):
            if len(rel.parts) == 1:
                lines.append(f"{rel}/  ({sum(len(f) for _, _, f in os.walk(dirpath))} files, not listed)")
            dirnames[:] = []
            continue
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            lines.append(f"{(rel / fn)}  {p.stat().st_size:,} B")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"…[{len(lines) - max_lines} more]"]
    return "\n".join(lines) or "(empty)"


# ---------------------------------------------------------------- resuming after limits / new chat
def resume_packet(decisions_tail: int = 200) -> str:
    """Everything a NEW chat needs to continue: charter + protocol + STATE.md + decisions.md + listing."""
    out = _out()
    state = (out / "STATE.md").read_text() if (out / "STATE.md").exists() else "(STATE.md does not exist yet)"
    dec_p = out / "decisions.md"
    if dec_p.exists():
        dl = dec_p.read_text().splitlines()
        dec = "\n".join(dl if len(dl) <= decisions_tail else [f"…[first {len(dl) - decisions_tail} lines omitted]"] + dl[-decisions_tail:])
    else:
        dec = "(decisions.md does not exist yet)"
    resume = _prompt_text_file("00c_resume.md")
    resume = resume.replace("<<<STATE>>>", state).replace("<<<DECISIONS>>>", dec).replace("<<<LISTING>>>", _listing(out))
    preamble = _prompt_text("preamble").split("\nNow acknowledge")[0].rstrip()   # no "wait for Phase 1" when resuming
    text = _prompt_text(0) + "\n\n---\n\n" + preamble + "\n\n---\n\n" + resume
    (out / "logs" / f"resume_{_dt.datetime.now():%Y%m%d_%H%M%S}.md").write_text(text)
    note("resume packet built", f"{len(text):,} chars")
    print(PASTE_START)
    print(text)
    print(PASTE_END)
    if len(text) > 60_000:
        print(f"\nWARNING: {len(text):,} characters may exceed the chat's message limit. "
              f"Call apd.resume_packet(decisions_tail=80) for a shorter one.")
    return text


def _prompt_text_file(name: str) -> str:
    return (PROMPTS / name).read_text().replace("<your_name>", _model())


# ---------------------------------------------------------------- output contract
CONTRACT = {
    "predictions.csv": {"session_id", "risk_score", "risk_label"},
    "train_cv_predictions.csv": {"session_id", "fold", "risk_score", "risk_label"},
    "channels.csv": {"session_id", "vision_score", "language_score", "acoustic_score", "fused_score", "affect_labels"},
    "affect_features.csv": {"session_id", "cohort"},
    "cv_results.json": None, "failures.csv": {"session_id", "stage", "reason"}, "budget.csv": None,
    "audit_notes.md": None, "proposal.md": None, "phase3_summary.md": None, "strata_report.md": None,
    "affect_analysis.md": None, "methods.md": None, "decisions.md": None, "README.md": None,
    "STATE.md": None, "run.sh": None, "requirements.txt": None,
}


def check_contract(out_dir: Optional[Path] = None, quiet: bool = False) -> dict[str, str]:
    """Which contract files exist and have the required columns. Also checks predictions cover the test set."""
    out = Path(out_dir) if out_dir else _out()
    res = {}
    for name, cols in CONTRACT.items():
        p = out / name
        if not p.exists():
            res[name] = "MISSING"
            continue
        if cols:
            with p.open(newline="") as f:
                header = set(next(csv.reader(f), []))
            miss = cols - header
            res[name] = "ok" if not miss else f"missing columns {sorted(miss)}"
        else:
            res[name] = "ok"
    pred = out / "predictions.csv"
    test_dir = REPO / "data" / "test"
    if pred.exists() and test_dir.exists():
        with pred.open(newline="") as f:
            ids = {r["session_id"] for r in csv.DictReader(f)}
        test_ids = {p.stem for p in test_dir.glob("*") if p.suffix.lower() in _VIDEO_EXT}
        if test_ids:
            res["predictions coverage"] = f"{len(ids & test_ids)}/{len(test_ids)} test sessions"
    if not quiet:
        print(PASTE_START)
        for k, v in res.items():
            print(f"{k:28s} {v}")
        print(PASTE_END)
    return res


# ---------------------------------------------------------------- coordinator
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _script(name: str) -> str:
    return str(REPO / "scripts" / name)


def coordinator_dry_run(participant_regex: str = r"^([^_]+)_", n_show: int = 8) -> None:
    """Look at the source folder and show how files would be grouped into participants. Copies nothing."""
    mount()
    src = find_source_folder()
    if src is None:
        raise FileNotFoundError(
            "Lab 'video' folder not found. In Drive open 'Shared with me' > video > 'Add shortcut to Drive' > "
            f"My Drive, then re-run. (Folder id {SOURCE_FOLDER_ID}.)")
    subprocess.run([sys.executable, _script("make_split.py"), "--src", str(src), "--out", "/nonexistent",
                    "--participant-regex", participant_regex, "--dry-run", "--show", str(n_show)], check=True)


def coordinator_split(test_frac: float = 0.3, seed: int = 0, participant_regex: str = r"^([^_]+)_") -> Path:
    mount()
    src = find_source_folder()
    if src is None:
        raise FileNotFoundError("Lab 'video' folder not found; run coordinator_dry_run() for instructions.")
    out = BENCH / "data"
    PRIVATE.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, _script("make_split.py"), "--src", str(src), "--out", str(out),
                    "--private-out", str(PRIVATE), "--test-frac", str(test_frac), "--seed", str(seed),
                    "--participant-regex", participant_regex], check=True)
    return out


def coordinator_audit() -> Path:
    mount()
    data = BENCH / "data"
    subprocess.run([sys.executable, _script("audit_videos.py"), "--data", str(data), "--out", str(data / "audit.csv")], check=True)
    return data / "audit.csv"


def coordinator_publish_code() -> Path:
    """Copy this repo (code + prompts only) to MyDrive/apd_benchmark/code/research for operators without GitHub access."""
    dest = BENCH / "code" / "research"
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns(".git", "data", "outputs", "cache", "runs", "results", "__pycache__", "*.egg-info", ".venv")
    shutil.copytree(REPO, dest, ignore=ignore, symlinks=False)
    print(f"code copied to {dest}")
    return dest


def coordinator_preflight() -> list[str]:
    """Checks before sharing apd_benchmark with operators."""
    data = BENCH / "data"
    issues = []
    for need in ("train/Experimental", "train/Control", "train/train_labels.csv", "test", "audit.csv"):
        if not (data / need).exists():
            issues.append(f"missing {data / need}")
    for p in data.rglob("*"):
        if any(n in p.name for n in LEAK_NAMES):
            issues.append(f"private file inside shared data: {p}")
    test_sub = [p for p in (data / "test").iterdir() if p.is_dir()] if (data / "test").exists() else []
    if test_sub:
        issues.append(f"test/ has sub-folders (could reveal labels): {[p.name for p in test_sub]}")
    if not (PRIVATE / "test_labels.csv").exists():
        issues.append(f"{PRIVATE / 'test_labels.csv'} missing — scoring will be impossible")
    print("PREFLIGHT:", "OK — share MyDrive/apd_benchmark (Viewer) with the operators." if not issues else "")
    for i in issues:
        print("  -", i)
    return issues


def coordinator_collect(models=MODELS) -> Path:
    """Copy each operator's shared apd_outputs_<model> folder into repo outputs/<model>/ for scoring."""
    mount()
    dest_root = REPO / "outputs"
    dest_root.mkdir(exist_ok=True)
    for m in models:
        cands = [MYDRIVE / f"apd_outputs_{m}", *_shortcut_targets(f"apd_outputs_{m}")]
        src = next((c for c in cands if (c / "predictions.csv").exists()), None)
        if src is None:
            print(f"{m:7s} not found (ask the operator to share apd_outputs_{m} and add a shortcut)")
            continue
        dest = dest_root / m
        if dest.is_symlink():
            dest.unlink()
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("cache"))
        print(f"{m:7s} collected from {src}")
    return dest_root


def coordinator_score(labels: Optional[Path] = None) -> int:
    labels = Path(labels) if labels else PRIVATE / "test_labels.csv"
    data = BENCH / "data"
    return subprocess.run([sys.executable, _script("score_submissions.py"), "--outputs", str(REPO / "outputs"),
                           "--labels", str(labels), "--train-labels", str(data / "train" / "train_labels.csv"),
                           "--audit", str(data / "audit.csv"), "--out", str(REPO / "results" / "arm_b")]).returncode
