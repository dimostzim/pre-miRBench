import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402


def test_pinned_image_accepts_only_the_image_recorded_in_image_ids(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "image_ids.tsv").write_text("# comment\ntool\timage_id\tbuilt\nmirdnn\tsha256:aaa\t2026-07-04\n")
    local_images = {"mirdnn:latest": "sha256:aaa"}
    monkeypatch.setattr(train.subprocess, "run", lambda cmd, **kwargs: types.SimpleNamespace(stdout=local_images.get(cmd[-1], "") + "\n"))
    assert train.pinned_image(str(tmp_path), "mirdnn") == "mirdnn:latest"
    local_images["mirdnn:latest"] = "sha256:bbb"  # e.g. an older build tagged latest
    with pytest.raises(SystemExit, match="expects sha256:aaa"):
        train.pinned_image(str(tmp_path), "mirdnn")
    del local_images["mirdnn:latest"]
    with pytest.raises(SystemExit, match="not present"):
        train.pinned_image(str(tmp_path), "mirdnn")
