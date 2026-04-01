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

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to triage a realistic email inbox — classifying urgency, prioritizing workload, and drafting professional replies.

Built for the [Meta PyTorch × Scaler SST Hackathon](https://scaler.com/school-of-technology/meta-pytorch-hackathon).

---

## What this environment does

The agent receives a simulated inbox of 5 emails per episode. Each email is drawn from a pool containing urgent incidents, normal workplace messages, and spam. The agent must process each email one at a time.

**Three tasks of increasing difficulty:**

| Task | Difficulty | What the agent must do | Reward |
|---|---|---|---|
| `classify` | Easy | Label each email: `urgent`, `normal`, or `spam` | 1.0 = correct, 0.0 = wrong |
| `prioritize` | Medium | Classify + assign priority 1-5 | 0.5 category + 0.5 priority proximity |
| `reply` | Hard | Classify + prioritize + write a reply | 0.4 + 0.2 + 0.4 reply quality |

Rewards are always in **[0.0, 1.0]** and give partial credit — the agent gets something for being close even when not perfect.

---

## Quick start

### Connect to the live HF Space

```python
import urllib.request, json

# Reset
req = urllib.request.Request(
    "https://Deathblue1306-email-triage-env.hf.space/reset",
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)
obs = json.loads(urllib.request.urlopen(req).read().decode())
print(obs["observation"]["current_email"]["subject"])

# Step
data = json.dumps({"action": {"category": "urgent", "priority": 1, "reply": "On it."}}).encode()
req = urllib.request.Request(
    "https://Deathblue1306-email-triage-env.hf.space/step",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
result = json.loads(urllib.request.urlopen(req).read().decode())
print(result["reward"])
```

### Run locally with Docker

```bash
# Build the image
docker build -t email-triage-env -f server/Dockerfile .

# Run the server
docker run -p 7860:7860 email-triage-env
```

### Run the baseline inference script

The baseline uses the OpenAI client pointed at any OpenAI-compatible endpoint.
We used [Groq](https://console.groq.com) (free, no billing needed) with `llama-3.1-8b-instant`.

```bash
# Required environment variables
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set HF_TOKEN=gsk_YOUR_GROQ_KEY
set ENV_BASE_URL=https://Deathblue1306-email-triage-env.hf.space

python inference.py
```

The script uses the official OpenAI Python client (`from openai import OpenAI`) with `base_url=API_BASE_URL` — compatible with OpenAI, Groq, Gemini, or any OpenAI-compatible endpoint.

---

## Action space

```python
class EmailAction(Action):
    category: str    # "urgent" | "normal" | "spam"
    priority: int    # 1 (most urgent) to 5 (least urgent), default 3
    reply: str       # draft reply text, default ""
```

## Observation space

```python
class EmailObservation(Observation):
    current_email: Email          # the email to process
    next_email_subject: str       # preview of what's coming next
    feedback: str                 # how the last action was graded
    emails_remaining: int         # how many emails are left
    done: bool                    # True when inbox is empty (inherited)
    reward: float                 # reward for the last action (inherited)
```

---

## Project structure

```
Email_Triage/
├── models.py                 # Action, Observation, State definitions
├── client.py                 # EmailTriageEnv HTTP client
├── inference.py              # LLM baseline script (OpenAI client)
├── openenv.yaml              # Environment manifest
├── pyproject.toml            # Package config
├── README.md                 # This file
└── server/
    ├── __init__.py
    ├── email_environment.py  # Core logic: reset(), step(), state(), graders
    ├── app.py                # FastAPI wiring
    ├── Dockerfile            # Container definition
    └── requirements.txt      # Server dependencies
```

---

## Reward function design

The reward function gives **partial credit** throughout the trajectory, not just at the end. This is important for RL training because the agent gets a learning signal on every step.

- **Classify task**: Binary (1.0 / 0.0). Simple, unambiguous ground truth.
- **Prioritize task**: Category score (0.5) + priority proximity score (0.5). A priority of 2 when the answer is 1 earns 0.375/0.5, not zero.
- **Reply task**: Category (0.4) + priority (0.2) + reply quality (0.4). Reply quality uses keyword overlap with an ideal reply + length heuristic.

---

## Baseline scores

Baseline agent: `llama-3.1-8b-instant` via Groq, using the OpenAI client with `base_url=https://api.groq.com/openai/v1`.

| Task | Model | Score (avg over 3 episodes) |
|---|---|---|
| classify | llama-3.1-8b-instant | 0.837 |
| prioritize | llama-3.1-8b-instant | 0.892 |
| reply | llama-3.1-8b-instant | 0.651 |

---

## HF Space

Live at: [https://huggingface.co/spaces/Deathblue1306/email-triage-env](https://huggingface.co/spaces/Deathblue1306/email-triage-env)
