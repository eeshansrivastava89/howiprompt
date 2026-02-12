"""Data models for the How I Prompt analytics pipeline."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Platform(str, Enum):
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"


class Role(str, Enum):
    HUMAN = "human"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Unified message from any source."""
    timestamp: datetime
    platform: Platform
    role: Role
    content: str
    conversation_id: str
    word_count: int
    model_id: str | None = None
    model_provider: str | None = None


class PersonaType(str, Enum):
    """Exactly 4 personas based on 2x2 matrix."""
    COLLABORATOR = "collaborator"   # High engagement, High politeness
    EXPLORER = "explorer"           # High engagement, Low politeness
    EFFICIENT = "efficient"         # Low engagement, High politeness
    PRAGMATIST = "pragmatist"       # Low engagement, Low politeness


@dataclass
class PersonaDefinition:
    """Definition of a persona type."""
    name: str
    description: str
    traits: list[str]


# === Persona Definitions (Exactly 4) ===
PERSONAS: dict[PersonaType, PersonaDefinition] = {
    PersonaType.COLLABORATOR: PersonaDefinition(
        name="The Collaborator",
        description="You ask questions politely. AI is your partner, not your tool.",
        traits=["Inquisitive", "Courteous", "Partnership-oriented"]
    ),
    PersonaType.EXPLORER: PersonaDefinition(
        name="The Explorer",
        description="You question, iterate, and dig deeper. Thinking out loud.",
        traits=["Curious", "Iterative", "Thorough"]
    ),
    PersonaType.EFFICIENT: PersonaDefinition(
        name="The Efficient",
        description="Polite but focused. You know what you want and ask nicely.",
        traits=["Respectful", "Direct", "Purposeful"]
    ),
    PersonaType.PRAGMATIST: PersonaDefinition(
        name="The Pragmatist",
        description="Balanced and practical. No frills, just results.",
        traits=["Balanced", "Practical", "Focused"]
    ),
}

SOURCE_VIEW_LABELS = {
    "both": "Claude Code + Codex",
    "claude_code": "Claude Code",
    "codex": "Codex",
}
