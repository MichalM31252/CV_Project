"""Cross-platform implementations of the shell-dependent Makefile targets.

GNU Make picks its own shell. On macOS and Linux that is ``/bin/sh``; on Windows
it is ``cmd.exe`` unless a POSIX ``sh`` happens to be on PATH. Recipes written
with ``grep``, ``awk``, ``rm -rf`` or a ``VAR=value command`` prefix therefore
work for some contributors and fail for others - which makes the documented entry
points in the README unreliable exactly where a reviewer is most likely to try
them.

Python is already a hard dependency of this project, so the portable move is to
implement those few targets here and have the Makefile shell out to Python.
Environment variables are set by Make itself via ``export``, which needs no shell
support at all.

    python scripts/devtools.py help | clean | deploy-info
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Generated artifacts, relative to the repository root. Globs are resolved
# against ROOT so `make clean` behaves the same from any working directory.
#
# reports/ is deliberately NOT cleaned. Those files are committed - the README
# links to reports/model_report.md as the evidence for its results table - so
# deleting them here would leave a broken link in the repository for anyone who
# ran `make clean` without re-running the pipeline. They are overwritten on the
# next training run anyway.
CLEAN_GLOBS = (
    "data/raw/*.parquet",
    "data/processed/*",
    "data/models/*",
)
CLEAN_DIRS = (".pytest_cache", ".ruff_cache")
# Never delete these - they are what keeps the empty directories in git.
KEEP = {".gitkeep"}


def cmd_help() -> int:
    """Print each Makefile target that carries a `## description` comment."""
    makefile = ROOT / "Makefile"
    pattern = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*):.*?## (.*)$", re.MULTILINE)
    for target, description in pattern.findall(makefile.read_text(encoding="utf-8")):
        print(f"  {target:<16} {description.strip()}")
    return 0


def cmd_clean() -> int:
    """Remove generated data, models, reports and caches."""
    removed = 0

    for pattern in CLEAN_GLOBS:
        for path in ROOT.glob(pattern):
            if path.name in KEEP:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1

    for name in CLEAN_DIRS:
        target = ROOT / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed += 1

    # Bytecode caches, wherever they are.
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1

    print(f"removed {removed} generated paths")
    return 0


DEPLOY_STEPS = """\
GCP deployment sequence
-----------------------
1. cd terraform && cp terraform.tfvars.example terraform.tfvars
   (set project_id, then review the plan)
2. terraform init && terraform apply
3. gcloud auth configure-docker $(terraform output -raw artifact_registry | cut -d/ -f1)
4. docker build -f docker/Dockerfile -t $(terraform output -raw artifact_registry)/api:v1 .
5. docker push $(terraform output -raw artifact_registry)/api:v1
6. terraform apply          # rolls the service onto the new image
7. curl $(terraform output -raw service_url)/health

Note: steps 3-5 use command substitution, which is bash syntax. In PowerShell use
      $(...) as-is but replace `cut -d/ -f1` with `.Split('/')[0]`, or run these
      steps from Git Bash / WSL.
"""


def cmd_deploy_info() -> int:
    print(DEPLOY_STEPS)
    return 0


COMMANDS = {"help": cmd_help, "clean": cmd_clean, "deploy-info": cmd_deploy_info}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    return COMMANDS[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
