"""2x2 persona classification algorithm."""

from .config import Config
from .models import PERSONAS, PersonaType


def classify_persona(
    politeness: float,
    backtrack: float,
    question_rate: float,
    command_rate: float,
    config: Config
) -> dict:
    """
    Classify user into exactly 1 of 4 personas using 2x2 matrix.

    Axes:
      X: Engagement Score = (question_rate + backtrack) / 2
         High engagement = asks questions, backtracks/iterates

      Y: Politeness Score = politeness - (command_rate * 0.5)
         High politeness = says please/thanks, fewer direct commands

    Quadrants:
      High engagement + High politeness -> Collaborator
      High engagement + Low politeness  -> Explorer
      Low engagement + High politeness  -> Efficient
      Low engagement + Low politeness   -> Pragmatist
    """
    # Compute composite scores
    engagement_score = (question_rate + backtrack) / 2
    politeness_score = politeness - (command_rate * 0.5)

    # Classify based on thresholds
    high_engagement = engagement_score >= config.persona_engagement_threshold
    high_politeness = politeness_score >= config.persona_politeness_threshold

    if high_engagement and high_politeness:
        persona_type = PersonaType.COLLABORATOR
    elif high_engagement and not high_politeness:
        persona_type = PersonaType.EXPLORER
    elif not high_engagement and high_politeness:
        persona_type = PersonaType.EFFICIENT
    else:
        persona_type = PersonaType.PRAGMATIST

    persona = PERSONAS[persona_type]

    return {
        "type": persona_type.value,
        "name": persona.name,
        "description": persona.description,
        "traits": persona.traits,
        "quadrant": {
            "engagement_score": round(engagement_score, 1),
            "politeness_score": round(politeness_score, 1),
            "high_engagement": high_engagement,
            "high_politeness": high_politeness,
        },
        "scores": {
            "politeness": round(politeness, 1),
            "backtrack": round(backtrack, 1),
            "question_rate": round(question_rate, 1),
            "command_rate": round(command_rate, 1),
        }
    }
