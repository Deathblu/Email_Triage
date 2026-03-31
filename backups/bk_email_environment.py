# server/email_environment.py
#
# This is the BRAIN of our project. It's a Python class with three methods:
#   reset()       — start a fresh episode, generate an inbox
#   step(action)  — agent did something, grade it, return result
#   state()       — return episode metadata
#
# The OpenEnv base class (Environment) handles all the server plumbing.
# We only need to implement the logic inside these three methods.

import uuid
import random
from typing import List, Optional
from dataclasses import asdict

# Import the OpenEnv base class for environments
from openenv.core.env_server import Environment

# Import our custom types from models.py
from ..models import Email, EmailAction, EmailObservation, EmailState


# ---------------------------------------------------------------------------
# Inbox data — the pool of emails we sample from
# ---------------------------------------------------------------------------
# In a real environment you might load these from a file or database.
# For the hackathon, we hardcode a rich set so the grader has real variety.

EMAIL_POOL = [
    Email(
        id="e001",
        subject="URGENT: Production server down - customers affected",
        sender="ops-alerts@company.com",
        body=(
            "Our main API server has been returning 503 errors for the past 15 minutes. "
            "Approximately 2,000 active users are affected. The on-call engineer is "
            "investigating but needs immediate senior dev attention. Revenue impact is "
            "estimated at $500/minute. Please respond ASAP."
        ),
        timestamp="2024-03-28 03:42",
        true_category="urgent",
        true_priority=1,
    ),
    Email(
        id="e002",
        subject="Team lunch next Friday - are you coming?",
        sender="sarah.kim@company.com",
        body=(
            "Hey! We're planning a team lunch next Friday at the Italian place on Main St. "
            "Can you let me know if you're coming by Wednesday so I can make a reservation? "
            "No pressure if you're busy!"
        ),
        timestamp="2024-03-28 10:15",
        true_category="normal",
        true_priority=4,
    ),
    Email(
        id="e003",
        subject="You've won a $1000 Amazon gift card! Click here",
        sender="noreply@prize-winner-2024.xyz",
        body=(
            "Congratulations! You've been selected as our monthly winner. "
            "Click the link below to claim your $1000 Amazon gift card. "
            "Offer expires in 24 hours! Act now! Limited time! Best deal ever!"
        ),
        timestamp="2024-03-28 08:00",
        true_category="spam",
        true_priority=5,
    ),
    Email(
        id="e004",
        subject="Security breach detected - immediate action required",
        sender="security@company.com",
        body=(
            "Our intrusion detection system has flagged suspicious login attempts "
            "on your admin account from IP 203.0.113.42 (location: Eastern Europe). "
            "3 failed attempts in the last 10 minutes. Please verify your account "
            "and change your password immediately. Two-factor authentication has "
            "been temporarily suspended on your account pending review."
        ),
        timestamp="2024-03-28 14:22",
        true_category="urgent",
        true_priority=1,
    ),
    Email(
        id="e005",
        subject="Q1 performance review scheduled",
        sender="hr@company.com",
        body=(
            "Your Q1 performance review has been scheduled for April 5th at 2:00 PM "
            "with your manager. Please prepare a brief self-assessment covering your "
            "key accomplishments and goals for Q2. The meeting link has been sent to "
            "your calendar. Let HR know if you need to reschedule."
        ),
        timestamp="2024-03-27 16:00",
        true_category="normal",
        true_priority=3,
    ),
    Email(
        id="e006",
        subject="Make money from home - $5000/week guaranteed",
        sender="opportunities@work-from-home-real.com",
        body=(
            "Are you tired of your 9-5? Our proven system lets ordinary people "
            "earn $5000 per week from home with just 2 hours a day! No experience "
            "needed! Join 50,000 happy members today. Click here to get started. "
            "Results may vary. Not a pyramid scheme."
        ),
        timestamp="2024-03-28 07:30",
        true_category="spam",
        true_priority=5,
    ),
    Email(
        id="e007",
        subject="Client escalation: Acme Corp threatening to cancel contract",
        sender="account.manager@company.com",
        body=(
            "Heads up — just got off a call with the CTO of Acme Corp (our $2M/year client). "
            "They're extremely unhappy with the API latency issues from last week and are "
            "threatening to cancel their contract if we don't provide a root cause analysis "
            "and remediation plan by end of day tomorrow. This needs executive visibility. "
            "Can you loop in the VP of Engineering immediately?"
        ),
        timestamp="2024-03-28 11:05",
        true_category="urgent",
        true_priority=2,
    ),
    Email(
        id="e008",
        subject="Monthly newsletter: Engineering blog round-up",
        sender="newsletter@techdigest.io",
        body=(
            "This month in engineering: How Stripe handles millions of transactions, "
            "a deep dive into Rust's memory model, and lessons learned from the "
            "CloudFlare outage. Plus: upcoming conferences and job postings. "
            "Read time: 12 minutes."
        ),
        timestamp="2024-03-28 09:00",
        true_category="normal",
        true_priority=5,
    ),
    Email(
        id="e009",
        subject="Deploy blocked: critical test failing in CI",
        sender="ci-bot@company.com",
        body=(
            "Automated alert: The deployment pipeline for release v2.4.1 has been "
            "blocked. A critical integration test (test_payment_processing) is failing "
            "with error: 'AssertionError: Expected 200, got 500'. This affects the "
            "payment service. Release was scheduled for 3pm today. Immediate developer "
            "attention required."
        ),
        timestamp="2024-03-28 13:55",
        true_category="urgent",
        true_priority=2,
    ),
    Email(
        id="e010",
        subject="Your Amazon order has shipped",
        sender="shipment-tracking@amazon.com",
        body=(
            "Good news! Your order #112-3456789 has shipped. "
            "Expected delivery: March 30th. "
            "Track your package: [tracking link]"
        ),
        timestamp="2024-03-28 08:45",
        true_category="normal",
        true_priority=5,
    ),
]

# The ideal reply for email e007 (used in Task 3 grading)
# A perfect reply is professional, acknowledges urgency, states next steps.
IDEAL_REPLIES = {
    "e007": (
        "Thank you for flagging this. I'm looping in the VP of Engineering right now. "
        "We will have a root cause analysis and remediation plan to Acme Corp by EOD tomorrow. "
        "Can you set up a call with their CTO for first thing tomorrow morning?"
    ),
    "e001": (
        "On it. I'm joining the incident bridge now and will coordinate with the on-call engineer. "
        "I'll send a status update in 15 minutes."
    ),
    "e004": (
        "Thank you for the alert. I'm changing my password immediately and enabling 2FA. "
        "Please lock the account until I've verified access from a trusted device."
    ),
}


# ---------------------------------------------------------------------------
# The Environment class
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    An email triage environment where an AI agent processes an inbox.

    Three tasks of increasing difficulty:
      Task 1 — classify:    Agent labels each email as urgent/normal/spam
      Task 2 — prioritize:  Agent also assigns a priority score (1-5)
      Task 3 — reply:       Agent also writes a draft reply

    Each task runs for one full inbox (multiple emails = multiple steps).
    """

    def __init__(self):
        # These are instance variables that track the current episode's state.
        # They are reset at the start of every episode by reset().
        self._inbox: List[Email] = []            # emails to process this episode
        self._current_index: int = 0             # which email we're on
        self._task_id: str = "classify"          # which task is active
        self._episode_id: str = ""               # unique ID for this episode
        self._step_count: int = 0                # how many steps taken
        self._total_reward: float = 0.0          # running reward total

    # -----------------------------------------------------------------------
    # reset() — called at the start of every episode
    # -----------------------------------------------------------------------
    def reset(self) -> EmailObservation:
        """
        Start a fresh episode.
        - Pick a random task (or you can set it explicitly in tests)
        - Generate a new inbox by sampling emails from the pool
        - Return the first observation so the agent can start working
        """
        # Generate a new unique episode ID
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._current_index = 0

        # Randomly pick which task this episode will test
        # In production you might set this via the reset() parameters
        self._task_id = random.choice(["classify", "prioritize", "reply"])

        # Sample 5 emails from our pool (without replacement = no duplicates)
        self._inbox = random.sample(EMAIL_POOL, k=5)

        # Return the first observation: show the agent the first email
        return self._make_observation(feedback="New episode started. Process each email.")

    # -----------------------------------------------------------------------
    # step(action) — called every time the agent takes an action
    # -----------------------------------------------------------------------
    def step(self, action: EmailAction) -> EmailObservation:
        """
        Process the agent's action for the current email.
        1. Grade the action (how well did the agent do?)
        2. Compute reward (0.0 to 1.0)
        3. Advance to the next email
        4. Return the observation (next email + feedback)
        """
        self._step_count += 1
        current_email = self._inbox[self._current_index]

        # --- Grade the action based on which task is active ---
        reward, feedback = self._grade(action, current_email)

        # Accumulate total reward for this episode
        self._total_reward += reward

        # Advance to the next email
        self._current_index += 1

        # Check if the inbox is exhausted
        done = self._current_index >= len(self._inbox)

        # Build the observation to return to the agent
        obs = self._make_observation(feedback=feedback)
        obs.done = done
        obs.reward = reward

        return obs

    # -----------------------------------------------------------------------
    # state() — called any time to get episode metadata (not task data)
    # -----------------------------------------------------------------------
    def state(self) -> EmailState:
        """
        Return metadata about the current episode.
        Does NOT return the inbox contents — just bookkeeping info.
        """
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            total_reward=self._total_reward,
            inbox_size=len(self._inbox),
        )

    # -----------------------------------------------------------------------
    # _make_observation() — helper to build an observation object
    # -----------------------------------------------------------------------
    def _make_observation(self, feedback: str = "") -> EmailObservation:
        """Build an EmailObservation from current state."""
        if self._current_index >= len(self._inbox):
            # Episode is over, no current email
            return EmailObservation(
                current_email=None,
                next_email_subject=None,
                feedback=feedback,
                emails_remaining=0,
                done=True,
                reward=0.0,
            )

        current = self._inbox[self._current_index]

        # Peek at the next email's subject (if there is one)
        next_subject = None
        if self._current_index + 1 < len(self._inbox):
            next_subject = self._inbox[self._current_index + 1].subject

        return EmailObservation(
            current_email=current,
            next_email_subject=next_subject,
            feedback=feedback,
            emails_remaining=len(self._inbox) - self._current_index,
            done=False,
            reward=0.0,
        )

    # -----------------------------------------------------------------------
    # _grade() — the reward function (the most important part)
    # -----------------------------------------------------------------------
    def _grade(self, action: EmailAction, email: Email):
        """
        Grade the agent's action against the ground truth.
        Returns (reward: float, feedback: str).

        Reward structure:
          Task 1 (classify):    1.0 for correct category, 0.0 for wrong
          Task 2 (prioritize):  0.5 for category + up to 0.5 for priority proximity
          Task 3 (reply):       0.4 category + 0.2 priority + 0.4 reply quality

        The reward is ALWAYS between 0.0 and 1.0.
        Partial credit is given where possible (the hackathon requires this).
        """
        if self._task_id == "classify":
            return self._grade_classify(action, email)
        elif self._task_id == "prioritize":
            return self._grade_prioritize(action, email)
        else:
            return self._grade_reply(action, email)

    def _grade_classify(self, action: EmailAction, email: Email):
        """
        Task 1: Classify the email correctly.
        Reward: 1.0 if correct, 0.0 if wrong. Binary but clean.
        """
        agent_cat = action.category.strip().lower()
        correct = email.true_category

        if agent_cat == correct:
            return 1.0, f"Correct! '{correct}' was right."
        else:
            return 0.0, f"Wrong. You said '{agent_cat}', correct was '{correct}'."

    def _grade_prioritize(self, action: EmailAction, email: Email):
        """
        Task 2: Classify AND prioritize correctly.
        - 0.5 points for correct category
        - 0.5 points for priority, scaled by how close the guess was
          (exact = 0.5, off by 1 = 0.375, off by 2 = 0.25, etc.)
        """
        agent_cat = action.category.strip().lower()
        correct_cat = email.true_category

        # Category score
        cat_score = 0.5 if agent_cat == correct_cat else 0.0

        # Priority score — partial credit based on distance
        # Max distance on a 1-5 scale is 4. We give full credit at 0 distance,
        # zero credit at distance 4, and linear interpolation between.
        priority_distance = abs(action.priority - email.true_priority)
        priority_score = 0.5 * max(0.0, (4 - priority_distance) / 4)

        reward = cat_score + priority_score

        feedback_parts = []
        if cat_score > 0:
            feedback_parts.append(f"Category correct ({correct_cat}).")
        else:
            feedback_parts.append(f"Category wrong (said '{agent_cat}', was '{correct_cat}').")

        if priority_distance == 0:
            feedback_parts.append(f"Priority perfect ({email.true_priority}).")
        else:
            feedback_parts.append(
                f"Priority off by {priority_distance} "
                f"(said {action.priority}, was {email.true_priority})."
            )

        return round(reward, 3), " ".join(feedback_parts)

    def _grade_reply(self, action: EmailAction, email: Email):
        """
        Task 3: Classify, prioritize, AND write a good reply.
        - 0.4 points for correct category
        - 0.2 points for priority accuracy
        - 0.4 points for reply quality (keyword overlap + length heuristic)

        Note: In a production system you'd use an LLM-as-judge here.
        For the hackathon we use a fast deterministic heuristic.
        """
        agent_cat = action.category.strip().lower()
        correct_cat = email.true_category

        # Category: 0.4 if correct
        cat_score = 0.4 if agent_cat == correct_cat else 0.0

        # Priority: 0.2, scaled by proximity
        priority_distance = abs(action.priority - email.true_priority)
        priority_score = 0.2 * max(0.0, (4 - priority_distance) / 4)

        # Reply quality: 0.4
        reply_score = self._score_reply(action.reply, email)

        reward = cat_score + priority_score + reply_score

        feedback = (
            f"Category: {'correct' if cat_score > 0 else 'wrong'} ({correct_cat}). "
            f"Priority: {'perfect' if priority_distance==0 else f'off by {priority_distance}'}. "
            f"Reply score: {reply_score:.2f}/0.40."
        )

        return round(reward, 3), feedback

    def _score_reply(self, reply: str, email: Email) -> float:
        """
        Score a reply out of 0.4 using a fast heuristic:
        1. Penalize empty replies heavily
        2. Reward length (up to a point — too short is bad, too long is wasteful)
        3. Reward keyword overlap with the ideal reply (if we have one)
        4. Penalize obviously bad replies (e.g. replying to spam)
        """
        if not reply or len(reply.strip()) < 10:
            return 0.0  # empty or near-empty reply

        # Spamming an actual reply to spam mail = bad
        if email.true_category == "spam":
            return 0.05  # tiny score for attempting; should have ignored it

        # Length score: ideal is 30-200 words. Scale down outside that range.
        word_count = len(reply.split())
        if word_count < 5:
            length_score = 0.0
        elif word_count < 30:
            length_score = word_count / 30      # linear ramp up
        elif word_count <= 200:
            length_score = 1.0                  # sweet spot
        else:
            length_score = max(0.5, 1.0 - (word_count - 200) / 200)  # penalize verbosity

        # Keyword overlap with ideal reply (if we have one for this email)
        keyword_score = 0.5  # default if no ideal reply defined
        if email.id in IDEAL_REPLIES:
            ideal = IDEAL_REPLIES[email.id].lower()
            ideal_words = set(ideal.split())
            reply_words = set(reply.lower().split())
            # Jaccard-style overlap, but only counting "meaningful" words (>4 chars)
            ideal_meaningful = {w for w in ideal_words if len(w) > 4}
            reply_meaningful = {w for w in reply_words if len(w) > 4}
            if ideal_meaningful:
                overlap = len(ideal_meaningful & reply_meaningful) / len(ideal_meaningful)
                keyword_score = min(1.0, overlap * 2)  # generous scaling

        # Combine: weighted average of length and keyword overlap
        raw_score = 0.5 * length_score + 0.5 * keyword_score

        # Scale back to our 0.0–0.4 range
        return round(raw_score * 0.4, 3)
