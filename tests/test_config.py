from pathlib import Path

from pathsnap.config import REPO_ROOT, load_config


def test_default_config_paths_are_absolute_and_under_repo():
    cfg = load_config()
    assert cfg["seed"] == 42
    for p in cfg["paths"].values():
        assert p.is_absolute()
        assert REPO_ROOT in p.parents


def test_custom_config_file(tmp_path: Path):
    f = tmp_path / "c.yaml"
    f.write_text("seed: 7\npaths:\n  results: out\n")
    cfg = load_config(f)
    assert cfg["seed"] == 7
    assert cfg["paths"]["results"] == (REPO_ROOT / "out").resolve()
