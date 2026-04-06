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
from models import Email, PublicEmail, EmailAction, EmailObservation, EmailState


# ---------------------------------------------------------------------------
# Email pool — 30 emails covering urgent, normal, spam, ambiguous,
# Indian corporate context, fake-urgent, and follow-up scenarios
# ---------------------------------------------------------------------------

EMAIL_POOL = [
    # --- Original 10 ---
    Email(id="e001", subject="URGENT: Production server down - customers affected",
          sender="ops-alerts@company.com", timestamp="2024-03-28 03:42",
          body="Our main API server has been returning 503 errors for the past 15 minutes. "
               "Approximately 2,000 active users are affected. Revenue impact is $500/minute. Please respond ASAP.",
          true_category="urgent", true_priority=1,
          true_routing="escalate_team", true_team="engineering"),

    Email(id="e002", subject="Team lunch next Friday - are you coming?",
          sender="sarah.kim@company.com", timestamp="2024-03-28 10:15",
          body="Hey! We're planning a team lunch next Friday at the Italian place. "
               "Can you let me know by Wednesday so I can make a reservation?",
          true_category="normal", true_priority=4,
          true_routing="handle", true_team="none"),

    Email(id="e003", subject="You've won a $1000 Amazon gift card! Click here",
          sender="noreply@prize-winner-2024.xyz", timestamp="2024-03-28 08:00",
          body="Congratulations! You've been selected as our monthly winner. "
               "Click the link below to claim your $1000 Amazon gift card. Act now!",
          true_category="spam", true_priority=5,
          true_routing="archive", true_team="none"),

    Email(id="e004", subject="Security breach detected - immediate action required",
          sender="security@company.com", timestamp="2024-03-28 14:22",
          body="Our intrusion detection system flagged suspicious login attempts on your admin account "
               "from IP 203.0.113.42. 3 failed attempts in 10 minutes. Please change your password immediately.",
          true_category="urgent", true_priority=1,
          true_routing="escalate_manager", true_team="none"),

    Email(id="e005", subject="Q1 performance review scheduled",
          sender="hr@company.com", timestamp="2024-03-27 16:00",
          body="Your Q1 performance review has been scheduled for April 5th at 2:00 PM with your manager. "
               "Please prepare a brief self-assessment. Let HR know if you need to reschedule.",
          true_category="normal", true_priority=3,
          true_routing="handle", true_team="none"),

    Email(id="e006", subject="Make money from home - $5000/week guaranteed",
          sender="opportunities@work-from-home-real.com", timestamp="2024-03-28 07:30",
          body="Are you tired of your 9-5? Our proven system lets ordinary people earn $5000 per week "
               "from home with just 2 hours a day! No experience needed!",
          true_category="spam", true_priority=5,
          true_routing="archive", true_team="none"),

    Email(id="e007", subject="Client escalation: Acme Corp threatening to cancel contract",
          sender="account.manager@company.com", timestamp="2024-03-28 11:05",
          body="Just got off a call with Acme Corp CTO. They're extremely unhappy with API latency issues "
               "and threatening to cancel their $2M/year contract if we don't provide RCA by end of day tomorrow.",
          true_category="urgent", true_priority=2,
          true_routing="escalate_manager", true_team="none"),

    Email(id="e008", subject="Monthly newsletter: Engineering blog round-up",
          sender="newsletter@techdigest.io", timestamp="2024-03-28 09:00",
          body="This month: How Stripe handles millions of transactions, a deep dive into Rust's memory model, "
               "and lessons from the CloudFlare outage. Read time: 12 minutes.",
          true_category="normal", true_priority=5,
          true_routing="handle", true_team="none"),

    Email(id="e009", subject="Deploy blocked: critical test failing in CI",
          sender="ci-bot@company.com", timestamp="2024-03-28 13:55",
          body="Deployment pipeline for release v2.4.1 is blocked. Critical integration test "
               "(test_payment_processing) failing: AssertionError: Expected 200, got 500. Release scheduled for 3pm today.",
          true_category="urgent", true_priority=2,
          true_routing="escalate_team", true_team="engineering"),

    Email(id="e010", subject="Your Amazon order has shipped",
          sender="shipment-tracking@amazon.com", timestamp="2024-03-28 08:45",
          body="Good news! Your order #112-3456789 has shipped. Expected delivery: March 30th.",
          true_category="normal", true_priority=5,
          true_routing="handle", true_team="none"),

    # --- New 20 ---
    Email(id="e011", subject="Series A funding round closed - announcement inside",
          sender="ceo@startupco.in", timestamp="2024-03-28 09:30",
          body="Team, I'm thrilled to share that we've successfully closed our Series A round of $8M led by "
               "Sequoia Capital India. This is a huge milestone. All-hands meeting tomorrow at 10am to discuss "
               "what this means for our roadmap and hiring plans.",
          true_category="normal", true_priority=2,
          true_routing="handle", true_team="none"),

    Email(id="e012", subject="JIRA ticket SRE-4521 escalated to P0",
          sender="jira-bot@company.com", timestamp="2024-03-28 02:15",
          body="Ticket SRE-4521 has been escalated to P0 by the on-call engineer. "
               "Issue: Memory leak in the payment microservice causing OOM crashes every 45 minutes. "
               "Customer-facing impact: checkout failures. Assigned to your team. Immediate action required.",
          true_category="urgent", true_priority=1,
          true_routing="escalate_team", true_team="engineering"),

    Email(id="e013", subject="Leave application approved - Rahul Sharma",
          sender="hr-system@company.com", timestamp="2024-03-27 17:00",
          body="This is to inform you that the leave application submitted by Rahul Sharma "
               "(Employee ID: EMP-2341) for April 10-14 has been approved by the reporting manager. "
               "Please update the project plan accordingly.",
          true_category="normal", true_priority=4,
          true_routing="handle", true_team="none"),

    Email(id="e014", subject="Congratulations! You qualify for a pre-approved personal loan",
          sender="offers@quickloan-instant.co", timestamp="2024-03-28 06:00",
          body="Dear Customer, you have been pre-approved for a personal loan of up to Rs. 5,00,000 "
               "at just 0.1% interest! No documents needed. Click here to claim in 5 minutes. Limited offer!",
          true_category="spam", true_priority=5,
          true_routing="archive", true_team="none"),

    Email(id="e015", subject="Database migration window - your approval needed",
          sender="dba-team@company.com", timestamp="2024-03-28 12:00",
          body="Hi, we need your sign-off on the planned database migration for the user_events table. "
               "Proposed window: Saturday 2am-4am IST. Expected downtime: 90 minutes. "
               "This affects the analytics dashboard only — no customer-facing impact. Please reply with approval or concerns.",
          true_category="normal", true_priority=3,
          true_routing="handle", true_team="none"),

    Email(id="e016", subject="CRITICAL: Data breach - customer PII exposed",
          sender="security-alert@company.com", timestamp="2024-03-28 01:30",
          body="Our security monitoring detected an unauthorised export of the customers table at 01:14 IST. "
               "Approximately 50,000 customer records including names, emails, and phone numbers may have been "
               "exfiltrated. Legal and compliance teams have been notified. Need CTO and CEO on a call in 15 minutes.",
          true_category="urgent", true_priority=1,
          true_routing="escalate_manager", true_team="none"),

    Email(id="e017", subject="Re: Project Falcon - timeline update",
          sender="priya.menon@company.com", timestamp="2024-03-28 11:30",
          body="Hey, following up on our discussion yesterday. The design team has finished the mockups "
               "and dev estimates are in. We're looking at a 3-week delay due to the API dependency on the "
               "payments team. Can we sync this week to reprioritize features and adjust the client timeline?",
          true_category="normal", true_priority=3,
          true_routing="handle", true_team="none"),

    Email(id="e018", subject="Free iPhone 15 Pro - Flipkart lucky draw winner!",
          sender="winner-notification@flipkart-rewards.net", timestamp="2024-03-28 07:00",
          body="Congratulations! Your mobile number was selected in Flipkart's anniversary lucky draw. "
               "You have won an iPhone 15 Pro worth Rs. 1,34,900! Claim within 24 hours by clicking the link "
               "and paying a small processing fee of Rs. 499.",
          true_category="spam", true_priority=5,
          true_routing="archive", true_team="none"),

    Email(id="e019", subject="Vendor invoice overdue - Infosys BPO",
          sender="accounts@company.com", timestamp="2024-03-28 10:00",
          body="This is a reminder that the invoice INV-2024-0892 from Infosys BPO for Rs. 14,50,000 "
               "is now 45 days overdue. Infosys has sent a final notice and threatened to pause services "
               "by end of this week if payment is not received. Finance team needs your approval to release the payment.",
          true_category="urgent", true_priority=2,
          true_routing="escalate_team", true_team="finance"),

    Email(id="e020", subject="New joinee onboarding - please complete buddy assignment",
          sender="hr@company.com", timestamp="2024-03-27 15:00",
          body="We have 3 new engineers joining next Monday — Aditya Kumar, Sneha Patel, and Rohan Verma. "
               "As per our buddy program, could you please confirm which team members will be assigned as "
               "buddies? Please respond by Thursday so we can share the plan with the new joinees.",
          true_category="normal", true_priority=4,
          true_routing="handle", true_team="none"),

    Email(id="e021", subject="IMPORTANT: Your account will be suspended in 24 hours",
          sender="support@amazon-security-alert.com", timestamp="2024-03-28 08:30",
          body="We have detected unusual activity on your Amazon account. Your account will be permanently "
               "suspended unless you verify your identity immediately. Click here to verify: http://amaz0n-verify.ru/login",
          true_category="spam", true_priority=5,
          true_routing="archive", true_team="none"),

    Email(id="e022", subject="Kubernetes cluster OOM - pods restarting repeatedly",
          sender="monitoring@company.com", timestamp="2024-03-28 04:10",
          body="Alert: 8 out of 12 pods in the recommendation-service namespace are in CrashLoopBackOff. "
               "Root cause appears to be a memory leak introduced in the v3.2.0 deployment 2 hours ago. "
               "The recommendation service is degraded — users see empty product recommendations. On-call engineer pinged.",
          true_category="urgent", true_priority=1,
          true_routing="escalate_team", true_team="engineering"),

    Email(id="e023", subject="Hackathon participation - team registration open",
          sender="events@hasgeek.com", timestamp="2024-03-27 12:00",
          body="The IN/Clojure 2024 hackathon registrations are now open! Build something interesting "
               "over a weekend with your team. Prizes worth Rs. 2,00,000. Registration deadline: April 5th. "
               "Food and accommodation provided for outstation participants.",
          true_category="normal", true_priority=5,
          true_routing="handle", true_team="none"),

    Email(id="e024", subject="Re: Salary revision - clarification needed",
          sender="ankit.gupta@company.com", timestamp="2024-03-28 09:45",
          body="Hi, I received my revised offer letter but there seems to be a discrepancy. "
               "The letter shows a 12% hike but my manager had communicated 18% during the appraisal discussion. "
               "Could you please check and clarify? This is time-sensitive as I need to respond to the letter by tomorrow.",
          true_category="normal", true_priority=2,
          true_routing="escalate_team", true_team="hr"),

    Email(id="e025", subject="AWS bill spike - $47,000 overage this month",
          sender="billing-alerts@company.com", timestamp="2024-03-28 00:05",
          body="Your AWS bill for March has reached $89,432 against a budget of $42,000. "
               "Primary drivers: EC2 instances in us-east-1 (untagged, possibly orphaned) and "
               "S3 data transfer costs from the new video feature. Immediate review needed to avoid "
               "further overage. FinOps team has flagged 23 idle instances for termination — need your approval.",
          true_category="urgent", true_priority=2,
          true_routing="handle", true_team="none"),

    Email(id="e026", subject="Referral bonus processed - thank you!",
          sender="hr@company.com", timestamp="2024-03-27 18:00",
          body="Hi, this is to confirm that your employee referral bonus of Rs. 50,000 for referring "
               "Priya Sharma (who joined last month) has been processed and will reflect in your April salary. "
               "Thank you for helping us grow the team!",
          true_category="normal", true_priority=5,
          true_routing="handle", true_team="none"),

    Email(id="e027", subject="Urgent: Sign NDA before 5pm today - legal requirement",
          sender="legal@company.com", timestamp="2024-03-28 11:00",
          body="As part of the upcoming acquisition discussions with TechCorp, all senior engineers "
               "are required to sign a mutual NDA before we share any technical documentation. "
               "The first meeting is at 6pm today. Please sign the attached document and return by 5pm. "
               "Contact legal@company.com if you have questions.",
          true_category="urgent", true_priority=2,
          true_routing="escalate_team", true_team="legal"),

    Email(id="e028", subject="Work from home policy update - new guidelines",
          sender="hr@company.com", timestamp="2024-03-27 14:00",
          body="Please find attached the updated Work From Home policy effective April 1st. "
               "Key changes: WFH days limited to 2 per week, core hours 11am-4pm IST are mandatory online, "
               "and all WFH requests beyond the standard 2 days need manager approval. "
               "Please read and acknowledge by March 31st.",
          true_category="normal", true_priority=3,
          true_routing="handle", true_team="none"),

    Email(id="e029", subject="Crypto investment opportunity - 300% returns guaranteed",
          sender="invest@cryptoprofits-india.xyz", timestamp="2024-03-28 05:30",
          body="Namaste! We are offering exclusive access to our AI-powered crypto trading bot "
               "that has delivered 300% returns in the last 6 months. Minimum investment: Rs. 10,000. "
               "Join 50,000 Indians who are already earning passive income. Limited slots available. "
               "WhatsApp us now: +91-XXXXXXXXXX",
          true_category="spam", true_priority=5,
          true_routing="archive", true_team="none"),

    Email(id="e030", subject="On-call handover notes - please review before 6pm",
          sender="vikram.nair@company.com", timestamp="2024-03-28 15:30",
          body="Hey, I'm handing over on-call to you at 6pm. Here's the current situation: "
               "1) The Redis cache is running at 87% capacity - keep an eye on it. "
               "2) There's a known flaky test in the auth service - ignore if it fires once, page me if twice. "
               "3) Infra team is doing a planned network maintenance at 2am - expect brief connectivity blip. "
               "Runbook links are in the #oncall Slack channel. Ping me if anything looks off.",
          true_category="normal", true_priority=2,
          true_routing="handle", true_team="none"),
]

IDEAL_REPLIES = {
    "e007": "Thank you for flagging this. I'm looping in the VP of Engineering now. "
            "We will have a root cause analysis to Acme Corp by EOD tomorrow.",
    "e001": "On it. I'm joining the incident bridge now and will send a status update in 15 minutes.",
    "e004": "Thank you for the alert. I'm changing my password immediately and enabling 2FA.",
    "e009": "Investigating now. I'll have a fix deployed before the 3pm release window.",
    "e012": "Acknowledged. I'm pulling in the team now. We'll roll back v3.2.0 immediately and investigate the memory leak.",
    "e016": "Understood. I'm joining the call in 15 minutes. Please also loop in our legal counsel and notify affected customers per GDPR guidelines.",
    "e019": "Approving the payment release. Please process immediately and send confirmation to Infosys BPO.",
    "e022": "Rolling back to v3.1.9 now. Will post an update in #incidents in 10 minutes.",
    "e025": "Please terminate the 23 idle instances immediately. I'll review the full bill breakdown today and set up budget alerts.",
    "e027": "Signing the NDA now and will return by 5pm. Please confirm receipt.",
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
        self._task_id = random.choice(["classify", "prioritize", "reply", "escalate"])
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
        """Build observation from current state.
        Strips ground truth labels via to_public() — agent never sees true_category etc.
        """
        if self._current_index >= len(self._inbox):
            return EmailObservation(
                current_email=None,
                next_email_subject=None,
                feedback=feedback,
                emails_remaining=0,
                task_id=self._task_id,
            )
        current = self._inbox[self._current_index]
        next_subject = (
            self._inbox[self._current_index + 1].subject
            if self._current_index + 1 < len(self._inbox) else None
        )
        return EmailObservation(
            current_email=current.to_public(),
            next_email_subject=next_subject,
            feedback=feedback,
            emails_remaining=len(self._inbox) - self._current_index,
            task_id=self._task_id,
        )

    # Partial credit table for wrong classifications.
    # Reflects that urgent↔normal is a closer mistake than urgent↔spam.
    CONFUSION_SCORE = {
        ("urgent", "normal"): 0.3,
        ("normal", "urgent"): 0.3,
        ("normal", "spam"):   0.1,
        ("spam", "normal"):   0.1,
        ("urgent", "spam"):   0.0,
        ("spam", "urgent"):   0.0,
    }

    # Professional language markers — reward formal, actionable replies
    PROFESSIONAL_MARKERS = [
        "dear", "hello", "hi", "thank you", "thanks", "regards",
        "sincerely", "please", "kindly", "appreciate", "noted",
    ]

    # Urgency acknowledgement words — reward replies that show awareness
    URGENCY_MARKERS = [
        "immediately", "right away", "on it", "urgent", "asap",
        "now", "priority", "escalat", "alert", "critical", "resolv",
    ]

    def _grade(self, action: EmailAction, email: Email):
        if self._task_id == "classify":
            return self._grade_classify(action, email)
        elif self._task_id == "prioritize":
            return self._grade_prioritize(action, email)
        elif self._task_id == "reply":
            return self._grade_reply(action, email)
        else:
            return self._grade_escalate(action, email)

    def _grade_escalate(self, action: EmailAction, email: Email):
        """
        Task 4: Escalate — decide who handles this email.

        Routing options:
          handle           — deal with it yourself
          escalate_manager — forward to your manager
          escalate_team    — forward to a specific team
          archive          — discard (spam / irrelevant)

        Reward structure:
          Correct routing + correct team → 1.0
          Correct routing, wrong/missing team → 0.6
          Adjacent mistake (escalate_manager vs escalate_team) → 0.3
          Archive a critical email (priority 1) → 0.0 hard penalty
          Handle when should escalate → 0.2
          Wrong routing for spam (not archived) → 0.1
        """
        predicted_routing = action.routing.strip().lower()
        predicted_team    = action.team.strip().lower()
        correct_routing   = email.true_routing
        correct_team      = email.true_team

        # Hard penalty — archiving a critical email is catastrophic
        if predicted_routing == "archive" and email.true_priority <= 1:
            return 0.0, f"Critical error! Archived a priority-1 email. Correct: {correct_routing}/{correct_team}."

        # Perfect match
        if predicted_routing == correct_routing:
            if correct_routing == "escalate_team":
                if predicted_team == correct_team:
                    return 1.0, f"Perfect! Routing: {correct_routing}, team: {correct_team}."
                else:
                    return 0.6, (
                        f"Routing correct ({correct_routing}) but wrong team. "
                        f"Said '{predicted_team}', should be '{correct_team}'."
                    )
            else:
                return 1.0, f"Perfect! Routing: {correct_routing}."

        # Adjacent mistakes — escalate_manager vs escalate_team
        adjacent = {
            ("escalate_manager", "escalate_team"),
            ("escalate_team", "escalate_manager"),
        }
        if (predicted_routing, correct_routing) in adjacent:
            return 0.3, (
                f"Close — said '{predicted_routing}', correct was '{correct_routing}'. "
                f"Partial credit: 0.3."
            )

        # Handling when should escalate
        if predicted_routing == "handle" and correct_routing in ("escalate_manager", "escalate_team"):
            return 0.2, f"Should have escalated (correct: {correct_routing}/{correct_team}) but chose to handle."

        # Not archiving spam
        if correct_routing == "archive" and predicted_routing != "archive":
            return 0.1, f"Should have archived this email but said '{predicted_routing}'."

        return 0.0, f"Wrong routing. Said '{predicted_routing}', correct was '{correct_routing}/{correct_team}'."

    def _grade_classify(self, action: EmailAction, email: Email):
        """
        Smarter classify grader — partial credit for adjacent mistakes.
        urgent↔normal = 0.3, normal↔spam = 0.1, urgent↔spam = 0.0
        """
        predicted = action.category.strip().lower()
        correct = email.true_category

        if predicted == correct:
            return 1.0, f"Correct! '{correct}' was right."

        # Partial credit based on confusion severity
        partial = self.CONFUSION_SCORE.get((predicted, correct), 0.0)
        if partial > 0:
            return partial, (
                f"Close — said '{predicted}', correct was '{correct}'. "
                f"Partial credit: {partial}."
            )
        return 0.0, f"Wrong. Said '{predicted}', correct was '{correct}'. No partial credit."

    def _grade_prioritize(self, action: EmailAction, email: Email):
        """
        Smarter prioritize grader:
        - Category: partial credit via confusion table (max 0.5)
        - Priority: exponential decay — small errors forgiven more than large ones
        """
        predicted = action.category.strip().lower()
        correct_cat = email.true_category

        # Category score — use confusion table for partial credit
        if predicted == correct_cat:
            cat_score = 0.5
            cat_msg = "Category correct."
        else:
            partial = self.CONFUSION_SCORE.get((predicted, correct_cat), 0.0)
            cat_score = partial * 0.5   # scale to 0.5 max
            cat_msg = f"Category wrong (was '{correct_cat}'), partial: {cat_score:.2f}."

        # Priority score — exponential decay instead of linear
        # off by 0 → 0.5, off by 1 → 0.47, off by 2 → 0.25, off by 4 → 0.03
        distance = abs(action.priority - email.true_priority)
        priority_score = round(0.5 * (0.5 ** distance), 3)
        pri_msg = "Priority perfect." if distance == 0 else f"Priority off by {distance} (score: {priority_score:.3f})."

        reward = round(cat_score + priority_score, 3)
        return reward, f"{cat_msg} {pri_msg}"

    def _grade_reply(self, action: EmailAction, email: Email):
        """
        Smarter reply grader:
        - Category: confusion table partial credit (max 0.4)
        - Priority: exponential decay (max 0.2)
        - Reply: 5 sub-scores — length, keywords, professionalism, urgency, structure
        - Bad behaviour penalties: replying to spam, too-short reply to critical email
        """
        predicted = action.category.strip().lower()
        correct_cat = email.true_category

        # Category score with partial credit
        if predicted == correct_cat:
            cat_score = 0.4
        else:
            partial = self.CONFUSION_SCORE.get((predicted, correct_cat), 0.0)
            cat_score = round(partial * 0.4, 3)

        # Priority score with exponential decay
        distance = abs(action.priority - email.true_priority)
        priority_score = round(0.2 * (0.5 ** distance), 3)

        # Reply score
        reply_score = self._score_reply(action.reply, email)

        reward = round(cat_score + priority_score + reply_score, 3)
        return reward, (
            f"Category: {'correct' if predicted == correct_cat else f'wrong (was {correct_cat}), partial={cat_score:.2f}'}. "
            f"Priority: {'perfect' if distance == 0 else f'off by {distance} ({priority_score:.2f})'}. "
            f"Reply: {reply_score:.2f}/0.40."
        )

    def _score_reply(self, reply: str, email: Email) -> float:
        """
        Multi-factor reply scorer — 5 sub-scores scaled to 0.0–0.4:
          1. Length       — appropriate length for the situation
          2. Keywords     — addresses actual email content
          3. Professional — uses professional language markers
          4. Urgency      — acknowledges urgency for critical emails
          5. Structure    — has greeting and clear action statement

        Bad behaviour penalties:
          - Replying to spam → 0.0
          - Critical email (priority 1) + reply under 20 words → 0.5x penalty
        """
        # Penalty: never reply to spam
        if email.true_category == "spam":
            return 0.0

        # Empty or trivial reply
        if not reply or len(reply.strip()) < 10:
            return 0.0

        reply_lower = reply.lower()
        words = reply.split()
        wc = len(words)

        # --- Sub-score 1: Length ---
        # Ideal range: 20–150 words. Ramp up below 20, penalise above 150.
        if wc < 5:
            length_score = 0.0
        elif wc < 20:
            length_score = wc / 20
        elif wc <= 150:
            length_score = 1.0
        else:
            length_score = max(0.4, 1.0 - (wc - 150) / 150)

        # --- Sub-score 2: Keyword overlap with ideal reply ---
        keyword_score = 0.5  # default when no ideal reply defined
        if email.id in IDEAL_REPLIES:
            ideal_words = {w for w in IDEAL_REPLIES[email.id].lower().split() if len(w) > 4}
            reply_words = {w for w in reply_lower.split() if len(w) > 4}
            if ideal_words:
                overlap = len(ideal_words & reply_words) / len(ideal_words)
                keyword_score = min(1.0, overlap * 2)

        # --- Sub-score 3: Professionalism ---
        # Count how many professional markers appear in the reply
        prof_hits = sum(1 for marker in self.PROFESSIONAL_MARKERS if marker in reply_lower)
        professional_score = min(1.0, prof_hits / 2)  # 2+ markers = full score

        # --- Sub-score 4: Urgency acknowledgement ---
        # Only matters for urgent/priority-1 emails
        if email.true_category == "urgent" or email.true_priority <= 2:
            urgency_hits = sum(1 for marker in self.URGENCY_MARKERS if marker in reply_lower)
            urgency_score = min(1.0, urgency_hits / 2)
        else:
            urgency_score = 1.0  # not urgent — full marks by default

        # --- Sub-score 5: Structure ---
        # Does the reply have a greeting AND a sentence that implies action?
        has_greeting = any(g in reply_lower for g in ["dear", "hello", "hi", "hey"])
        has_action = any(a in reply_lower for a in [
            "will", "shall", "going to", "i'll", "we'll", "let me",
            "on it", "handling", "investigating", "sending", "forwarding"
        ])
        structure_score = (0.5 if has_greeting else 0.0) + (0.5 if has_action else 0.0)

        # --- Combine sub-scores ---
        raw = (
            0.25 * length_score +
            0.25 * keyword_score +
            0.20 * professional_score +
            0.20 * urgency_score +
            0.10 * structure_score
        )

        # --- Bad behaviour penalty ---
        # Critical email (priority 1) + very short reply = not acceptable
        if email.true_priority == 1 and wc < 20:
            raw *= 0.5

        return round(raw * 0.4, 3)