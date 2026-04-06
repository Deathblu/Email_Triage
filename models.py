# models.py
# OpenEnv base classes are Pydantic BaseModel with extra="forbid".
# Our subclasses must ONLY add fields — no @dataclass, no extra kwargs.

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from openenv.core.env_server import Action, Observation, State


class PublicEmail(BaseModel):
    """Email as seen by the agent — ground truth labels stripped out."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class Email(BaseModel):
    """A single email with ground truth labels. Used internally by the grader."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    true_category: str
    true_priority: int
    true_routing: str = "handle"
    true_team: str = "none"

    def to_public(self) -> PublicEmail:
        """Strip ground truth — return only what the agent should see."""
        return PublicEmail(
            id=self.id,
            subject=self.subject,
            sender=self.sender,
            body=self.body,
            timestamp=self.timestamp,
        )


class EmailAction(Action):
    """
    What the agent sends each step.
    - category, priority, reply: used in classify / prioritize / reply tasks
    - routing, team: used in escalate task
    """
    category: str = "normal"            # urgent | normal | spam
    priority: int = 3                   # 1 (most urgent) to 5 (least urgent)
    reply: str = ""                     # draft reply — only graded in reply task
    routing: str = "handle"             # handle | escalate_manager | escalate_team | archive
    team: str = "none"                  # engineering | legal | finance | hr | none


class EmailObservation(Observation):
    """What the agent sees after reset() or step().
    Uses PublicEmail so ground truth labels are never exposed to the agent.
    """
    current_email: Optional[PublicEmail] = None
    next_email_subject: Optional[str] = None
    feedback: str = ""
    emails_remaining: int = 0
    task_id: str = "classify"


class EmailState(State):
    """Episode metadata returned by state()."""
    task_id: str = "classify"
    total_reward: float = 0.0
    inbox_size: int = 0