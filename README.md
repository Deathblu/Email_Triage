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
docker build -t email-triage-env -f server/Dockerfile .
docker run -p 7860:7860 email-triage-env
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

## Baseline scores

Baseline agent: `llama-3.1-8b-instant` via Groq, using the OpenAI client with `base_url=https://api.groq.com/openai/v1`.

| Task       | Model       | Score (avg over 3 episodes) |
| ---------- | ----------- | --------------------------- |
| classify   | gpt-4o-mini | —                           |
| prioritize | gpt-4o-mini | —                           |
| reply      | gpt-4o-mini | —                           |

_Run `baseline_inference.py` to populate these._

---

## HF Space

Live at: [https://huggingface.co/spaces/Deathblue1306/email-triage-env](https://huggingface.co/spaces/Deathblue1306/email-triage-env)
