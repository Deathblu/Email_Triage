# models.py
# OpenEnv base classes are Pydantic BaseModel with extra="forbid".
# Our subclasses must ONLY add fields — no @dataclass, no extra kwargs.

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from openenv.core.env_server import Action, Observation, State


class Email(BaseModel):
    """A single email. Plain Pydantic model — NOT an OpenEnv type."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    true_category: str
    true_priority: int


class EmailAction(Action):
    """What the agent sends each step.
    Only adds fields on top of Action base — no extra kwargs.
    """
    category: str
    priority: int = 3
    reply: str = ""


class EmailObservation(Observation):
    """What the agent sees after reset() or step().
    done and reward are inherited from Observation.
    current_email is Optional so it works when episode ends.
    """
    current_email: Optional[Email] = None
    next_email_subject: Optional[str] = None
    feedback: str = ""
    emails_remaining: int = 0


class EmailState(State):
    """Episode metadata returned by state()."""
    task_id: str = "classify"
    total_reward: float = 0.0
    inbox_size: int = 0