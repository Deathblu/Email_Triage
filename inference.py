# inference.py
# Baseline inference script for EmailTriageEnv.
# Uses raw HTTP calls directly — no client.py dependency.
#
# Required env vars:
#   API_BASE_URL  — LLM endpoint  e.g. https://api.openai.com/v1
#   MODEL_NAME    — e.g. gpt-4o-mini
#   HF_TOKEN      — your OpenAI or HF API key
#   ENV_BASE_URL  — your HF Space URL e.g. https://username-email-triage-env.hf.space

import os
import json
import urllib.request
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN", "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")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://Deathblue1306-email-triage-env.hf.space")

EPISODES_PER_TASK = 3
TASKS = ["classify", "prioritize", "reply"]


# ---------------------------------------------------------------------------
# Raw HTTP helpers — call the environment server directly
# ---------------------------------------------------------------------------

def env_reset():
    req = urllib.request.Request(
        f"{ENV_BASE_URL}/reset",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


def env_step(category: str, priority: int, reply: str):
    data = json.dumps({
        "action": {"category": category, "priority": priority, "reply": reply}
    }).encode()
    req = urllib.request.Request(
        f"{ENV_BASE_URL}/step",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def build_prompt(email: dict, task_id: str) -> str:
    task_instructions = {
        "classify": (
            "Classify this email as exactly one of: urgent, normal, spam.\n"
            "Respond ONLY with valid JSON: {\"category\": \"<urgent|normal|spam>\"}"
        ),
        "prioritize": (
            "Classify this email AND assign a priority.\n"
            "Category: urgent, normal, or spam.\n"
            "Priority: integer 1 (most urgent) to 5 (least urgent).\n"
            "Respond ONLY with valid JSON: {\"category\": \"...\", \"priority\": <1-5>}"
        ),
        "reply": (
            "Classify this email, assign a priority, AND write a professional reply.\n"
            "Category: urgent, normal, or spam.\n"
            "Priority: integer 1 (most urgent) to 5 (least urgent).\n"
            "For spam, leave reply empty.\n"
            "Respond ONLY with valid JSON: {\"category\": \"...\", \"priority\": <1-5>, \"reply\": \"...\"}"
        ),
    }
    return f"""You are an email triage assistant.

EMAIL:
Subject:  {email.get('subject', '')}
From:     {email.get('sender', '')}
Received: {email.get('timestamp', '')}
Body:
{email.get('body', '')}

TASK: {task_instructions[task_id]}"""


def parse_response(text: str, task_id: str):
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return {
            "category": str(data.get("category", "normal")).lower().strip(),
            "priority": int(data.get("priority", 3)),
            "reply":    str(data.get("reply", "")),
        }
    except Exception:
        print(f"    [warn] parse failed: {text[:80]}")
        return {"category": "normal", "priority": 3, "reply": ""}


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(llm_client, task_id: str) -> float:
    result = env_reset()
    obs = result.get("observation", {})
    total_reward = 0.0
    step = 0

    while True:
        email = obs.get("current_email")
        done  = result.get("done", False) or obs.get("done", False)

        if done or not email:
            break

        prompt   = build_prompt(email, task_id)
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        action = parse_response(response.choices[0].message.content, task_id)

        result = env_step(action["category"], action["priority"], action["reply"])
        reward = result.get("reward") or 0.0
        total_reward += reward
        step += 1

        obs      = result.get("observation", {})
        feedback = obs.get("feedback", "")
        print(f"    Step {step}: category={action['category']} priority={action['priority']} "
              f"reward={reward:.3f} | {feedback}")

        if result.get("done") or obs.get("done"):
            break

    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.")
        return

    print("EmailTriageEnv — Baseline Inference")
    print(f"  LLM:   {API_BASE_URL} / {MODEL_NAME}")
    print(f"  Env:   {ENV_BASE_URL}")
    print("=" * 60)

    llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    all_scores = {}

    for task_id in TASKS:
        print(f"\nTask: {task_id.upper()}")
        print("-" * 40)
        rewards = []
        for ep in range(EPISODES_PER_TASK):
            print(f"  Episode {ep+1}/{EPISODES_PER_TASK}:")
            total = run_episode(llm_client, task_id)
            normalized = total / 5.0
            rewards.append(normalized)
            print(f"  → Total: {total:.3f} | Normalized: {normalized:.3f}")

        avg = sum(rewards) / len(rewards)
        all_scores[task_id] = round(avg, 3)
        print(f"\n  Average for '{task_id}': {avg:.3f}")

    print("\n" + "=" * 60)
    print("BASELINE SCORES — copy into openenv.yaml:")
    for task_id, score in all_scores.items():
        print(f"  {task_id}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    main()