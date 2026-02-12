"""Data sync from local home directories into the project data folder."""

import shutil
from pathlib import Path


def copy_claude_code_data(dest_path: Path) -> bool:
    """Copy Claude Code data from ~/.claude/projects to data/claude_code/."""
    source_path = Path.home() / ".claude" / "projects"

    if not source_path.exists():
        print(f"  Error: Claude Code data not found at {source_path}")
        return False

    # Count files to copy
    files_to_copy = list(source_path.rglob("*.jsonl"))
    if not files_to_copy:
        print(f"  Error: No .jsonl files found in {source_path}")
        return False

    print(f"  Found {len(files_to_copy)} conversation files")

    # Copy directory structure
    for jsonl_file in files_to_copy:
        rel_path = jsonl_file.relative_to(source_path)
        dest_file = dest_path / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jsonl_file, dest_file)

    print(f"  Copied to {dest_path}")
    return True


def copy_codex_data(dest_path: Path) -> bool:
    """Copy Codex history from ~/.codex/history.jsonl to data/codex/history.jsonl."""
    source_path = Path.home() / ".codex" / "history.jsonl"

    if not source_path.exists():
        print(f"  Error: Codex history not found at {source_path}")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    print(f"  Copied to {dest_path}")
    return True
