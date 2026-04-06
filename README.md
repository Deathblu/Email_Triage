---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
tags:
  - openenv
pinned: false
---

# EmailTriageEnv

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to triage a realistic email inbox — classifying urgency, prioritizing workload, drafting professional replies, and making routing decisions.

Built for the [Meta PyTorch × Scaler SST Hackathon](https://scaler.com/school-of-technology/meta-pytorch-hackathon).

---

## What this environment does

The agent receives a simulated inbox of 5 emails per episode. Emails are sampled from a pool of 30 realistic messages covering urgent incidents, normal workplace communication, spam, Indian corporate scenarios, and ambiguous edge cases. The agent processes each email one at a time across four tasks of increasing difficulty.

**Four tasks:**

| Task         | Difficulty | What the agent must do                                              | Reward                                                    |
| ------------ | ---------- | ------------------------------------------------------------------- | --------------------------------------------------------- |
| `classify`   | Easy       | Label each email: `urgent`, `normal`, or `spam`                     | 1.0 correct, partial credit for adjacent mistakes         |
| `prioritize` | Medium     | Classify + assign priority 1–5                                      | 0.5 category + 0.5 exponential priority decay             |
| `reply`      | Hard       | Classify + prioritize + write a professional reply                  | 0.4 + 0.2 + 0.4 multi-factor reply score                  |
| `escalate`   | Hard       | Decide routing: handle / escalate_manager / escalate_team / archive | 1.0 correct routing + team, partial for adjacent mistakes |

Rewards are always in **[0.0, 1.0]**. Partial credit is given throughout the trajectory so the agent always has a learning signal.

---

## Reward function design

### Classify

Partial credit for adjacent mistakes — confusing `urgent` with `normal` (0.3) is penalised less than confusing `urgent` with `spam` (0.0). Binary rewards miss this nuance.

### Prioritize

Exponential decay instead of linear — off by 1 loses very little, off by 4 loses almost everything. Rewards closer guesses proportionally more.

### Reply

Five sub-scores combined:

- Length — appropriate for the situation
- Keyword overlap — addresses the actual email content
- Professionalism — uses formal language markers
- Urgency acknowledgement — shows awareness for critical emails
- Structure — has a greeting and a clear action statement

Bad behaviour penalties: replying to spam → 0.0. Critical email + reply under 20 words → 0.5x multiplier.

### Escalate

Rewards correct routing decisions with partial credit for near-misses. Hard penalty for archiving a priority-1 email (0.0 — catastrophic in a real workplace).

---

## Email pool

30 emails covering:

- Production outages, security breaches, CI failures, Kubernetes OOM
- Indian corporate scenarios: JIRA P0, vendor invoices, salary disputes, NDA signing, Series A announcements
- Spam: phishing, loan offers, crypto scams, fake prize draws
- Ambiguous emails that test judgment: on-call handovers, salary clarifications, AWS billing spikes

---

## Quick start

### Connect to the live HF Space

```python
import urllib.request, json

# Reset — start a new episode
req = urllib.request.Request(
    "https://Deathblue1306-email-triage-env.hf.space/reset",
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)
obs = json.loads(urllib.request.urlopen(req).read().decode())
print(obs["observation"]["task_id"])            # which task this episode
print(obs["observation"]["current_email"]["subject"])

# Step — send an action
data = json.dumps({"action": {
    "category": "urgent",
    "priority": 1,
    "reply": "On it — joining the incident bridge now.",
    "routing": "escalate_team",
    "team": "engineering"
}}).encode()
req = urllib.request.Request(
    "https://Deathblue1306-email-triage-env.hf.space/step",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
result = json.loads(urllib.request.urlopen(req).read().decode())
print(result["reward"])
print(result["observation"]["feedback"])
```

### Run locally with Docker

```bash
docker build -t email-triage-env -f server/Dockerfile .
docker run -p 7860:7860 email-triage-env
```

### Run the baseline inference script

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set HF_TOKEN=gsk_YOUR_GROQ_KEY
set ENV_BASE_URL=https://Deathblue1306-email-triage-env.hf.space
python inference.py
```

Uses the official OpenAI Python client with `base_url=API_BASE_URL` — compatible with OpenAI, Groq, Gemini, or any OpenAI-compatible endpoint.

---

## Action space

```python
class EmailAction(Action):
    category: str    # "urgent" | "normal" | "spam"
    priority: int    # 1 (most urgent) to 5 (least urgent), default 3
    reply: str       # draft reply — graded in reply task, default ""
    routing: str     # "handle" | "escalate_manager" | "escalate_team" | "archive"
    team: str        # "engineering" | "legal" | "finance" | "hr" | "none"
```

## Observation space

```python
class EmailObservation(Observation):
    current_email: PublicEmail    # email to process (ground truth labels hidden)
    next_email_subject: str       # subject preview of next email
    feedback: str                 # grader feedback on last action
    emails_remaining: int         # emails left in inbox
    task_id: str                  # active task: classify/prioritize/reply/escalate
    done: bool                    # True when inbox is empty (inherited)
    reward: float                 # reward for last action (inherited)
```

Ground truth labels (`true_category`, `true_priority`, `true_routing`, `true_team`) are never exposed to the agent — only the grader sees them.

---

## Project structure

```
Email_Triage/
├── models.py                 # Action, Observation, State, PublicEmail definitions
├── client.py                 # EmailTriageEnv HTTP client
├── inference.py              # LLM baseline script (OpenAI client)
├── openenv.yaml              # Environment manifest
├── pyproject.toml            # Package config
├── README.md                 # This file
└── server/
    ├── __init__.py
    ├── email_environment.py  # Core logic: reset(), step(), state(), all 4 graders
    ├── app.py                # FastAPI wiring
    ├── Dockerfile            # Container definition
    └── requirements.txt      # Server dependencies
```

---

## Baseline scores

Baseline agent: `llama-3.1-8b-instant` via Groq, using the OpenAI Python client.

| Task       | Model                | Score (avg over 3 episodes) |
| ---------- | -------------------- | --------------------------- |
| classify   | llama-3.1-8b-instant | 0.772                       |
| prioritize | llama-3.1-8b-instant | 0.667                       |
| reply      | llama-3.1-8b-instant | 0.678                       |
| escalate   | llama-3.1-8b-instant | 0.489                       |

---

## HF Space

Live at: [https://huggingface.co/spaces/Deathblue1306/email-triage-env](https://huggingface.co/spaces/Deathblue1306/email-triage-env)
