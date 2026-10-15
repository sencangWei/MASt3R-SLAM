import importlib.util
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/bind_runtime_lietorch.py"


def binding_module():
    spec = importlib.util.spec_from_file_location("runtime_binding", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency(tmp_path):
    source = tmp_path / "actual_lietorch"
    package = source / "lietorch/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("# source\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "lietorch"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "source"], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tool = tmp_path / "isolated"
    (tool / "thirdparty").mkdir(parents=True)
    return tool, source, package, commit


def test_missing_vendor_path_binds_to_actual_imported_git_source(tmp_path):
    tool, source, package, commit = dependency(tmp_path)
    report = binding_module().bind_runtime_lietorch(tool, package, commit)
    vendor = tool / "thirdparty/lietorch"
    assert vendor.is_symlink() and vendor.resolve() == source
    assert subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip() == commit
    assert report["commit"] == commit and report["source"] == str(source)
    assert binding_module().bind_runtime_lietorch(tool, package, commit) == report


@pytest.mark.parametrize("failure", ["wrong_commit", "dirty_source", "occupied", "parent_repo"])
def test_binding_fails_closed_without_replacing_paths(tmp_path, failure):
    tool, source, package, commit = dependency(tmp_path)
    vendor = tool / "thirdparty/lietorch"
    if failure == "wrong_commit":
        commit = "0" * 40
    elif failure == "dirty_source":
        package.write_text("# changed\n")
    elif failure == "occupied":
        vendor.mkdir()
    else:
        nested = source / "nested/lietorch/__init__.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("# not dependency root\n")
        package = nested
    with pytest.raises((ValueError, FileExistsError)):
        binding_module().bind_runtime_lietorch(tool, package, commit)
    assert not vendor.is_symlink()
