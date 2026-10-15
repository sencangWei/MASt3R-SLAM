"""Bind an isolated worktree's manifest path to its real, pinned runtime source."""

import hashlib
import json
from pathlib import Path
import subprocess


LIETORCH_COMMIT = "e7df86554156b36846008d8ddbcc4d8521a16554"


def bind_runtime_lietorch(tool, package_file, expected_commit=LIETORCH_COMMIT):
    tool = Path(tool).resolve(strict=True)
    package_file = Path(package_file).resolve(strict=True)
    source = package_file.parent.parent
    if package_file != source / "lietorch/__init__.py":
        raise ValueError("unexpected imported lietorch package layout")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError("imported lietorch does not have its own git identity")
    commit = git("rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"lietorch pin mismatch: expected {expected_commit}, imported {commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("imported lietorch has tracked source modifications")
    if git("ls-files", "--error-unmatch", "lietorch/__init__.py") != "lietorch/__init__.py":
        raise ValueError("imported package is not tracked by the pinned source")
    vendor = tool / "thirdparty/lietorch"
    if vendor.is_symlink():
        if vendor.resolve() != source:
            raise FileExistsError(f"refusing to replace existing binding: {vendor}")
    elif vendor.exists():
        raise FileExistsError(f"refusing to replace existing path: {vendor}")
    else:
        vendor.symlink_to(source, target_is_directory=True)
    return {
        "commit": commit,
        "source": str(source),
        "imported_package": str(package_file),
        "manifest_binding": str(vendor),
        "package_sha256": hashlib.sha256(package_file.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    import lietorch
    import lietorch_backends
    import lietorch_extras

    report = bind_runtime_lietorch(Path(__file__).resolve().parents[1], lietorch.__file__)
    report["native_extensions"] = {
        module.__name__: {
            "path": str(Path(module.__file__).resolve()),
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        }
        for module in (lietorch_backends, lietorch_extras)
    }
    print(json.dumps(report, indent=2))
