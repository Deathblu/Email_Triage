# server/email_environment.py
#
# Core environment logic. Three methods:
#   reset()  — start fresh episode, return first observation
#   step()   — grade action, return next observation with reward + done set
#   state()  — return episode metadata

import uuid
import random
from typing import List, Optional, Any

from openenv.core.env_server import Environment
from models import Email, EmailAction, EmailObservation, EmailState


# ---------------------------------------------------------------------------
# Email pool — 10 realistic emails the agent will be tested on
# ---------------------------------------------------------------------------

EMAIL_POOL = [
    Email(id="e001", subject="URGENT: Production server down - customers affected",
          sender="ops-alerts@company.com", timestamp="2024-03-28 03:42",
          body="Our main API server has been returning 503 errors for the past 15 minutes. "
               "Approximately 2,000 active users are affected. Revenue impact is $500/minute. Please respond ASAP.",
          true_category="urgent", true_priority=1),

    Email(id="e002", subject="Team lunch next Friday - are you coming?",
          sender="sarah.kim@company.com", timestamp="2024-03-28 10:15",
          body="Hey! We're planning a team lunch next Friday at the Italian place. "
               "Can you let me know by Wednesday so I can make a reservation?",
          true_category="normal", true_priority=4),

    Email(id="e003", subject="You've won a $1000 Amazon gift card! Click here",
          sender="noreply@prize-winner-2024.xyz", timestamp="2024-03-28 08:00",
          body="Congratulations! You've been selected as our monthly winner. "
               "Click the link below to claim your $1000 Amazon gift card. Act now!",
          true_category="spam", true_priority=5),

    Email(id="e004", subject="Security breach detected - immediate action required",
          sender="security@company.com", timestamp="2024-03-28 14:22",
          body="Our intrusion detection system flagged suspicious login attempts on your admin account "
               "from IP 203.0.113.42. 3 failed attempts in 10 minutes. Please change your password immediately.",
          true_category="urgent", true_priority=1),

    Email(id="e005", subject="Q1 performance review scheduled",
          sender="hr@company.com", timestamp="2024-03-27 16:00",
          body="Your Q1 performance review has been scheduled for April 5th at 2:00 PM with your manager. "
               "Please prepare a brief self-assessment. Let HR know if you need to reschedule.",
          true_category="normal", true_priority=3),

    Email(id="e006", subject="Make money from home - $5000/week guaranteed",
          sender="opportunities@work-from-home-real.com", timestamp="2024-03-28 07:30",
          body="Are you tired of your 9-5? Our proven system lets ordinary people earn $5000 per week "
               "from home with just 2 hours a day! No experience needed!",
          true_category="spam", true_priority=5),

    Email(id="e007", subject="Client escalation: Acme Corp threatening to cancel contract",
          sender="account.manager@company.com", timestamp="2024-03-28 11:05",
          body="Just got off a call with Acme Corp CTO. They're extremely unhappy with API latency issues "
               "and threatening to cancel their $2M/year contract if we don't provide RCA by end of day tomorrow.",
          true_category="urgent", true_priority=2),

    Email(id="e008", subject="Monthly newsletter: Engineering blog round-up",
          sender="newsletter@techdigest.io", timestamp="2024-03-28 09:00",
          body="This month: How Stripe handles millions of transactions, a deep dive into Rust's memory model, "
               "and lessons from the CloudFlare outage. Read time: 12 minutes.",
          true_category="normal", true_priority=5),

    Email(id="e009", subject="Deploy blocked: critical test failing in CI",
          sender="ci-bot@company.com", timestamp="2024-03-28 13:55",
          body="Deployment pipeline for release v2.4.1 is blocked. Critical integration test "
               "(test_payment_processing) failing: AssertionError: Expected 200, got 500. Release scheduled for 3pm today.",
          true_category="urgent", true_priority=2),

    Email(id="e010", subject="Your Amazon order has shipped",
          sender="shipment-tracking@amazon.com", timestamp="2024-03-28 08:45",
          body="Good news! Your order #112-3456789 has shipped. Expected delivery: March 30th.",
          true_category="normal", true_priority=5),
]

IDEAL_REPLIES = {
    "e007": "Thank you for flagging this. I'm looping in the VP of Engineering now. "
            "We will have a root cause analysis to Acme Corp by EOD tomorrow.",
    "e001": "On it. I'm joining the incident bridge now and will send a status update in 15 minutes.",
    "e004": "Thank you for the alert. I'm changing my password immediately and enabling 2FA.",
    "e009": "Investigating now. I'll have a fix deployed before the 3pm release window.",
}


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email triage environment. Three tasks:
      classify   — label each email: urgent / normal / spam
      prioritize — classify + assign priority 1-5
      reply      — classify + prioritize + write a reply
    """

    def __init__(self):
        super().__init__()
        self._inbox: List[Email] = []
        self._current_index: int = 0
        self._task_id: str = "classify"
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        """Start a fresh episode. Returns the first observation."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._current_index = 0
        self._task_id = random.choice(["classify", "prioritize", "reply"])
        self._inbox = random.sample(EMAIL_POOL, k=5)

        obs = self._make_observation(feedback="New episode started. Process each email.")
        obs.done = False
        obs.reward = None
        return obs

    def step(self, action: EmailAction, **kwargs: Any) -> EmailObservation:
        """Grade the action, advance to next email, return observation."""
        self._step_count += 1
        current_email = self._inbox[self._current_index]

        reward, feedback = self._grade(action, current_email)
        self._total_reward += reward
        self._current_index += 1

        done = self._current_index >= len(self._inbox)
        obs = self._make_observation(feedback=feedback)
        obs.reward = reward
        obs.done = done
        return obs

    @property
    def state(self) -> EmailState:
        """Return episode metadata."""
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            total_reward=self._total_reward,
            inbox_size=len(self._inbox),
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _make_observation(self, feedback: str = "") -> EmailObservation:
        """Build observation from current state."""
        if self._current_index >= len(self._inbox):
            return EmailObservation(
                current_email=None,
                next_email_subject=None,
                feedback=feedback,
                emails_remaining=0,
            )
        current = self._inbox[self._current_index]
        next_subject = (
            self._inbox[self._current_index + 1].subject
            if self._current_index + 1 < len(self._inbox) else None
        )
        return EmailObservation(
            current_email=current,
            next_email_subject=next_subject,
            feedback=feedback,
            emails_remaining=len(self._inbox) - self._current_index,
        )

    def _grade(self, action: EmailAction, email: Email):
        if self._task_id == "classify":
            return self._grade_classify(action, email)
        elif self._task_id == "prioritize":
            return self._grade_prioritize(action, email)
        else:
            return self._grade_reply(action, email)

    def _grade_classify(self, action: EmailAction, email: Email):
        correct = email.true_category
        if action.category.strip().lower() == correct:
            return 1.0, f"Correct! '{correct}' was right."
        return 0.0, f"Wrong. You said '{action.category}', correct was '{correct}'."

    def _grade_prioritize(self, action: EmailAction, email: Email):
        cat_score = 0.5 if action.category.strip().lower() == email.true_category else 0.0
        distance = abs(action.priority - email.true_priority)
        priority_score = 0.5 * max(0.0, (4 - distance) / 4)
        reward = cat_score + priority_score
        cat_msg = "Category correct." if cat_score > 0 else f"Category wrong (was '{email.true_category}')."
        pri_msg = "Priority perfect." if distance == 0 else f"Priority off by {distance}."
        return round(reward, 3), f"{cat_msg} {pri_msg}"

    def _grade_reply(self, action: EmailAction, email: Email):
        cat_score = 0.4 if action.category.strip().lower() == email.true_category else 0.0
        distance = abs(action.priority - email.true_priority)
        priority_score = 0.2 * max(0.0, (4 - distance) / 4)
        reply_score = self._score_reply(action.reply, email)
        reward = cat_score + priority_score + reply_score
        return round(reward, 3), (
            f"Category: {'correct' if cat_score > 0 else 'wrong'}. "
            f"Priority: {'perfect' if distance == 0 else f'off by {distance}'}. "
            f"Reply: {reply_score:.2f}/0.40."
        )

    def _score_reply(self, reply: str, email: Email) -> float:
        if not reply or len(reply.strip()) < 10:
            return 0.0
        if email.true_category == "spam":
            return 0.05
        words = reply.split()
        wc = len(words)
        if wc < 5:
            length_score = 0.0
        elif wc < 30:
            length_score = wc / 30
        elif wc <= 200:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (wc - 200) / 200)

        keyword_score = 0.5
        if email.id in IDEAL_REPLIES:
            ideal_words = {w for w in IDEAL_REPLIES[email.id].lower().split() if len(w) > 4}
            reply_words = {w for w in reply.lower().split() if len(w) > 4}
            if ideal_words:
                overlap = len(ideal_words & reply_words) / len(ideal_words)
                keyword_score = min(1.0, overlap * 2)

        return round((0.5 * length_score + 0.5 * keyword_score) * 0.4, 3)