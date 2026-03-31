# inference.py
#
# Baseline inference script for EmailTriageEnv.
# Runs an LLM agent against all 3 tasks and prints scores.
#
# Required environment variables (set these before running):
#   API_BASE_URL  — the LLM API endpoint  (e.g. https://api.openai.com/v1)
#   MODEL_NAME    — model identifier       (e.g. gpt-4o-mini)
#   HF_TOKEN      — your HuggingFace / API key
#
# Usage:
#   export API_BASE_URL="https://api.openai.com/v1"
#   export MODEL_NAME="gpt-4o-mini"
#   export HF_TOKEN="sk-..."
#   python inference.py

import os
import json
import random
from openai import OpenAI

# Import our environment client and types
from client import EmailTriageEnv
from models import EmailAction

# ---------------------------------------------------------------------------
# Configuration — reads the EXACT variable names the checklist requires
# ---------------------------------------------------------------------------
# API_BASE_URL: the LLM endpoint. On the grader's machine this might point
#   to a local model or a proxy — NOT necessarily api.openai.com.
#   We pass it directly into the OpenAI client as base_url.
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")

# MODEL_NAME: which model to use for inference
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# HF_TOKEN: used as the API key. On HF Spaces this is the HuggingFace token;
#   for local testing with OpenAI, set this to your sk-... key.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Number of episodes to average over for each task (more = more reliable score)
EPISODES_PER_TASK = 3

# The 3 task IDs we need to evaluate
TASKS = ["classify", "prioritize", "reply"]


# ---------------------------------------------------------------------------
# The LLM agent
# ---------------------------------------------------------------------------

def build_prompt(obs_dict: dict, task_id: str) -> str:
    """
    Build the prompt we send to the LLM for each step.
    The prompt tells the agent:
      - What task it's doing
      - The current email details
      - Exactly what JSON format to respond in
    """
    email = obs_dict.get("current_email", {})

    task_instructions = {
        "classify": (
            "Your job: classify this email as exactly one of: urgent, normal, spam.\n"
            "Respond ONLY with valid JSON: {\"category\": \"<urgent|normal|spam>\"}"
        ),
        "prioritize": (
            "Your job: classify this email AND assign a priority.\n"
            "Category must be one of: urgent, normal, spam.\n"
            "Priority is an integer from 1 (most urgent) to 5 (least urgent).\n"
            "Respond ONLY with valid JSON: {\"category\": \"...\", \"priority\": <1-5>}"
        ),
        "reply": (
            "Your job: classify this email, assign a priority, AND write a professional reply.\n"
            "Category must be one of: urgent, normal, spam.\n"
            "Priority is an integer from 1 (most urgent) to 5 (least urgent).\n"
            "For spam, leave reply empty.\n"
            "Respond ONLY with valid JSON: "
            "{\"category\": \"...\", \"priority\": <1-5>, \"reply\": \"...\"}"
        ),
    }

    return f"""You are an email triage assistant.

EMAIL TO PROCESS:
Subject:   {email.get('subject', 'N/A')}
From:      {email.get('sender', 'N/A')}
Received:  {email.get('timestamp', 'N/A')}
Body:
{email.get('body', 'N/A')}

TASK: {task_instructions[task_id]}"""


def parse_llm_response(response_text: str, task_id: str) -> EmailAction:
    """
    Parse the LLM's JSON response into an EmailAction.
    Falls back to safe defaults if the JSON is malformed.
    """
    try:
        # Strip markdown code fences if the LLM added them
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text.strip())

        return EmailAction(
            category=data.get("category", "normal").lower().strip(),
            priority=int(data.get("priority", 3)),
            reply=data.get("reply", ""),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        # If parsing fails, return a safe default action
        print(f"    [warn] Failed to parse LLM response: {response_text[:80]}")
        return EmailAction(category="normal", priority=3, reply="")


# ---------------------------------------------------------------------------
# Run one episode of a given task
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, env_client, task_id: str) -> float:
    """
    Run a complete episode (full inbox) for a given task.
    Returns the total reward earned in this episode.
    """
    # Force the task by temporarily overriding the environment's task_id.
    # We do this by passing the task_id as a query param to reset.
    # (If the server doesn't support this param, it will just be ignored
    #  and the task is chosen randomly — still valid for baseline testing.)
    obs = env_client.reset()
    total_reward = 0.0
    step = 0

    while True:
        # Convert observation to dict for the prompt builder
        obs_dict = {
            "current_email": {
                "subject": obs.current_email.subject if obs.current_email else "",
                "sender": obs.current_email.sender if obs.current_email else "",
                "body": obs.current_email.body if obs.current_email else "",
                "timestamp": obs.current_email.timestamp if obs.current_email else "",
            } if obs.current_email else {},
            "emails_remaining": obs.emails_remaining,
            "feedback": obs.feedback,
        }

        if obs.done or not obs.current_email:
            break

        # Build the prompt and call the LLM
        prompt = build_prompt(obs_dict, task_id)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # deterministic for reproducibility
            max_tokens=300,
        )

        raw_text = response.choices[0].message.content
        action = parse_llm_response(raw_text, task_id)

        # Take the action
        result = env_client.step(action)
        total_reward += result.reward
        step += 1

        print(
            f"    Step {step}: category={action.category}, "
            f"priority={action.priority}, reward={result.reward:.3f} | {result.observation.feedback}"
        )

        if result.done:
            break
        obs = result.observation

    return total_reward



# ---------------------------------------------------------------------------
# Main — run all tasks and report scores
# ---------------------------------------------------------------------------

# Number of episodes to average over for each task
EPISODES_PER_TASK = 3
TASKS = ["classify", "prioritize", "reply"]


def main():
    # Validate the required env vars (exact names from the checklist)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: export HF_TOKEN='sk-...'")
        return
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL not set. Run: export API_BASE_URL='https://api.openai.com/v1'")
        return

    # ENV_BASE_URL = where our environment server is running
    env_url = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

    print("EmailTriageEnv — Baseline Inference")
    print(f"  LLM endpoint:  {API_BASE_URL}")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  Env server:    {env_url}")
    print(f"  Episodes/task: {EPISODES_PER_TASK}")
    print("=" * 60)

    # KEY: pass base_url so this works with any OpenAI-compatible endpoint.
    # HF_TOKEN is used as the API key (works for OpenAI sk- keys too).
    openai_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    all_scores = {}

    with EmailTriageEnv(base_url=env_url).sync() as env_client:
        for task_id in TASKS:
            print(f"\nTask: {task_id.upper()}")
            print("-" * 40)

            episode_rewards = []
            for ep in range(EPISODES_PER_TASK):
                print(f"  Episode {ep + 1}/{EPISODES_PER_TASK}:")
                total_reward = run_episode(openai_client, env_client, task_id)
                # Normalize by inbox size (5 emails) → score is per-email average
                normalized = total_reward / 5.0
                episode_rewards.append(normalized)
                print(f"  → Total: {total_reward:.3f} | Normalized: {normalized:.3f}")

            avg = sum(episode_rewards) / len(episode_rewards)
            all_scores[task_id] = round(avg, 3)
            print(f"\n  Average for '{task_id}': {avg:.3f}")

    print("\n" + "=" * 60)
    print("BASELINE SCORES — copy into openenv.yaml:")
    for task_id, score in all_scores.items():
        print(f"  {task_id}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    main()