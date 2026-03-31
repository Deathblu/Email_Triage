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
from email_triage_env import EmailTriageEnv, EmailAction

with EmailTriageEnv(base_url="https://YOUR_HF_USERNAME-email-triage-env.hf.space").sync() as env:
    obs = env.reset()
    print(obs.current_email.subject)   # "URGENT: Production server down"

    result = env.step(EmailAction(
        category="urgent",
        priority=1,
        reply="On it — joining the incident bridge now."
    ))
    print(result.reward)    # e.g. 0.923
    print(result.observation.feedback)
```

### Run locally with Docker

```bash
# Build the image
docker build -t email-triage-env -f server/Dockerfile .

# Run the server
docker run -p 7860:7860 email-triage-env

# Connect to it
python -c "
from email_triage_env import EmailTriageEnv, EmailAction
with EmailTriageEnv(base_url='http://localhost:7860').sync() as env:
    obs = env.reset()
    print(obs.current_email.subject)
"
```

### Run the baseline

```bash
export OPENAI_API_KEY="sk-..."
export ENV_BASE_URL="http://localhost:7860"
python baseline_inference.py
```

---

## Action space

```python
@dataclass
class EmailAction(Action):
    category: str    # "urgent" | "normal" | "spam"
    priority: int    # 1 (most urgent) to 5 (least urgent), default 3
    reply: str       # draft reply text, default ""
```

## Observation space

```python
@dataclass
class EmailObservation(Observation):
    current_email: Email          # the email to process
    next_email_subject: str       # preview of what's coming next
    feedback: str                 # how the last action was graded
    emails_remaining: int         # how many emails are left
    done: bool                    # True when inbox is empty
    reward: float                 # reward for the last action
```

---

## Project structure

```
email_triage_env/
├── __init__.py               # Package exports
├── models.py                 # Action, Observation, State definitions
├── client.py                 # EmailTriageEnv client
├── baseline_inference.py     # LLM baseline script
├── openenv.yaml              # Environment manifest
├── pyproject.toml            # Package config
├── README.md                 # This file
└── server/
    ├── __init__.py
    ├── email_environment.py  # Core logic: reset(), step(), state(), graders
    ├── app.py                # FastAPI wiring (5 lines)
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

| Task | Model | Score (avg over 3 episodes) |
|---|---|---|
| classify | gpt-4o-mini | — |
| prioritize | gpt-4o-mini | — |
| reply | gpt-4o-mini | — |

*Run `baseline_inference.py` to populate these.*

---

## Setup & installation

```bash
# Install client
pip install git+https://huggingface.co/spaces/YOUR_HF_USERNAME/email-triage-env

# Install with dev deps (for running baseline)
pip install "email-triage-env[dev] @ git+https://huggingface.co/spaces/YOUR_HF_USERNAME/email-triage-env"
```

## Deploy to HF Spaces

```bash
pip install openenv-core huggingface_hub
huggingface-cli login
openenv push --repo-id YOUR_HF_USERNAME/email-triage-env
```
