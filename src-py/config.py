"""Build configuration, branding, and logging setup."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# Project root (where build.py lives)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("howiprompt")


@dataclass
class Config:
    """Build configuration - all paths relative to project root."""

    # Data sources - always read from data/ folder
    claude_code_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "claude_code")
    codex_history_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "codex" / "history.jsonl")
    codex_sessions_path: Path = field(default_factory=lambda: Path.home() / ".codex" / "sessions")

    # Output directory (relative to project root)
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    # Persona thresholds (calibrated from typical usage)
    # Engagement = (question_rate + backtrack) / 2
    # Politeness = politeness - (command_rate * 0.5)
    persona_engagement_threshold: float = 12.0
    persona_politeness_threshold: float = 4.5

    # Author info (removed in public version)
    author_domain: str = "eeshans.com"


def load_config() -> Config:
    """Load configuration. Could be extended to read from config.yaml."""
    return Config()


def load_branding() -> dict | None:
    """Load branding.json if it exists. Returns None if not found."""
    branding_path = PROJECT_ROOT / "branding.json"
    if branding_path.exists():
        with open(branding_path, 'r') as f:
            return json.load(f)
    return None
