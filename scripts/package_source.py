"""Package a clean, reviewed Git commit; never copy the working directory."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile


PRIVATE_COMPONENTS = {".git", ".venv", "__pycache__", "data", "runs", "work", "tmp", "dist"}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def package_source(repo: Path, output: Path, ref: str = "HEAD") -> dict:
    if git(repo, "status", "--porcelain").strip():
        raise ValueError("Commit and review intended changes first; the working tree must be clean.")
    commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    names = [name.decode() for name in git(repo, "ls-tree", "-r", "-z", "--name-only", commit).split(b"\0") if name]
    forbidden = [name for name in names if
                 PRIVATE_COMPONENTS.intersection(PurePosixPath(name).parts)
                 or PurePosixPath(name).name == ".env"
                 or PurePosixPath(name).suffix in {".pyc", ".pyo"}]
    if forbidden:
        raise ValueError("Refusing private or transient archive paths: " + ", ".join(forbidden))
    output = output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError("Choose a new archive name; existing releases are preserved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(["git", "-C", str(repo), "archive", "--format=zip",
                        "--prefix=weather-risk-forecaster/", f"--output={temporary}", commit], check=True)
        sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "commit": commit,
        "archive": output.name,
        "archive_sha256": sha256,
        "tracked_files": len(names),
        "scope": "Committed source, documentation and curated derived results; no private provider input bundle.",
        "method": "git archive",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD", help="Reviewed commit or tag (default: HEAD)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(package_source(root, args.output, args.ref), indent=2))


if __name__ == "__main__":
    main()
