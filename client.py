# client.py
#
# This is the CLIENT side of our environment — the code that agents (or you,
# manually) use to connect to the running server and call reset()/step()/state().
#
# Think of it like a TV remote:
#   - The TV (server) has the logic and state
#   - The remote (client) sends commands and receives responses
#
# OpenEnv's HTTPEnvClient base class handles the WebSocket connection,
# JSON serialization, and async/sync switching. We just teach it how to
# convert our specific JSON payloads into our typed Python objects.

from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult

# Import our custom types
from .models import EmailAction, EmailObservation, EmailState, Email


class EmailTriageEnv(HTTPEnvClient[EmailAction, EmailObservation]):
    """
    Client for the EmailTriageEnvironment server.

    Usage (sync):
        with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            result = env.step(EmailAction(category="urgent", priority=1))
            print(result.reward)

    Usage (async):
        async with EmailTriageEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(EmailAction(category="urgent", priority=1))
    """

    def _step_payload(self, action: EmailAction) -> dict:
        """
        Convert our EmailAction into a plain dictionary for JSON serialization.
        This is what gets sent over the wire to the server.

        The server's step() endpoint expects a JSON body like:
          {"category": "urgent", "priority": 1, "reply": "..."}
        """
        return {
            "category": action.category,
            "priority": action.priority,
            "reply": action.reply,
        }

    def _parse_result(self, payload: dict) -> StepResult[EmailObservation]:
        """
        Convert the JSON response from the server's /step endpoint
        back into our typed Python objects.

        The server returns JSON like:
          {
            "observation": { "current_email": {...}, "feedback": "...", ... },
            "reward": 0.75,
            "done": false
          }
        """
        # Parse the nested current_email dict back into an Email object
        obs_data = payload.get("observation", {})
        email_data = obs_data.get("current_email")

        current_email = None
        if email_data:
            current_email = Email(**email_data)

        obs = EmailObservation(
            current_email=current_email,
            next_email_subject=obs_data.get("next_email_subject"),
            feedback=obs_data.get("feedback", ""),
            emails_remaining=obs_data.get("emails_remaining", 0),
            done=obs_data.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> EmailState:
        """
        Convert the JSON response from /state into an EmailState object.
        """
        return EmailState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "classify"),
            total_reward=payload.get("total_reward", 0.0),
            inbox_size=payload.get("inbox_size", 0),
        )
