"""No-API protocol: split on a mounted Drive, isolation check, operator workspace, run/resume/contract."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fake_drive(tmp: Path) -> Path:
    d = tmp / "drive"
    for coh in ("Experimental", "Control"):
        for p in range(1, 6):
            for s in (1, 2):
                f = d / "MyDrive" / "video" / coh / f"{coh[0]}{p:02d}_s{s}.mp4"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"\x00" * (100 + p * 10 + s))
    return d


@pytest.fixture
def apd(tmp_path, monkeypatch):
    drive = _fake_drive(tmp_path)
    monkeypatch.setenv("APD_DRIVE_ROOT", str(drive))
    monkeypatch.setenv("APD_REPO", str(ROOT))
    import apd_bench.colab as m
    m = importlib.reload(m)
    yield m
    for model in m.MODELS:  # remove links the workspace created inside the repo
        link = ROOT / "outputs" / model
        if link.is_symlink():
            link.unlink()
    if (ROOT / "data").is_symlink():
        (ROOT / "data").unlink()
    if (ROOT / "outputs").is_dir() and not any((ROOT / "outputs").iterdir()):
        (ROOT / "outputs").rmdir()


def test_split_isolation_workspace(apd, tmp_path, monkeypatch, capsys):
    apd.coordinator_split(test_frac=0.4)
    data = apd.BENCH / "data"
    assert (data / "train" / "train_labels.csv").exists()
    assert not list(data.rglob("test_labels*")), "private labels must not be in shared data"
    assert (apd.PRIVATE / "test_labels.csv").exists()
    assert not [p for p in (data / "test").iterdir() if p.is_dir()]

    # operator on the same account before lock-down: refused
    with pytest.raises(RuntimeError):
        apd.setup_workspace("gpt")

    # lock-down: remove source shortcut and private folder from the operator's view
    (apd.MYDRIVE / "video").rename(tmp_path / "video_moved")
    apd.PRIVATE.rename(tmp_path / "private_moved")
    info = apd.setup_workspace("gpt")
    assert info["train_videos"] + info["test_videos"] == 20

    rc = apd.run(f"{sys.executable} -c \"[print('line', i) for i in range(300)]\"", live=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "lines omitted" in out and "COPY FROM HERE" in out

    (apd.REPO / "outputs" / "gpt" / "STATE.md").write_text("phase: 2\nnext command: x\n")
    text = apd.resume_packet()
    assert "phase: 2" in text and "<<<" not in text and "<your_name>" not in text
    assert "outputs/gpt/" in text and "wait for Phase 1" not in text

    res = apd.check_contract(quiet=True)
    assert res["predictions.csv"] == "MISSING" and res["STATE.md"] == "ok"
