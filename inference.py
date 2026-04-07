# inference.py
# EmailTriageEnv — OpenEnv-compliant inference script.
#
# Required env vars:
#   API_BASE_URL  — LLM endpoint         (default: Groq)
#   MODEL_NAME    — model identifier     (default: llama-3.1-8b-instant)
#   HF_TOKEN      — your Groq API key
#   ENV_BASE_URL  — your HF Space URL

import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://Deathblue1306-email-triage-env.hf.space")

EPISODES_PER_TASK  = 3
MAX_STEPS_PER_EP   = 10
TASKS              = ["classify", "prioritize", "reply", "escalate"]
BENCHMARK          = "email-triage-env"


# ---------------------------------------------------------------------------
# Mandatory stdout loggers  (validator parses these lines exactly)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    action_str = action.replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    try:
        req = urllib.request.Request(
            f"{ENV_BASE_URL}/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    except Exception as e:
        print(f"[DEBUG] env_reset failed: {e}", flush=True)
        return {"observation": {}, "done": True}


def env_step(category: str, priority: int, reply: str, routing: str = "handle", team: str = "none") -> dict:
    try:
        data = json.dumps({
            "action": {
                "category": category,
                "priority": priority,
                "reply":    reply,
                "routing":  routing,
                "team":     team,
            }
        }).encode()
        req = urllib.request.Request(
            f"{ENV_BASE_URL}/step",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    except Exception as e:
        print(f"[DEBUG] env_step failed: {e}", flush=True)
        return {"observation": {}, "reward": 0.0, "done": True}


# ---------------------------------------------------------------------------
# LLM helpers
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
            "IMPORTANT: Your reply text must be on a single line — use \\n for line breaks, not actual newlines.\n"
            "Respond ONLY with valid JSON: {\"category\": \"...\", \"priority\": <1-5>, \"reply\": \"...\"}"
        ),
        "escalate": (
            "Decide how to route this email. Choose routing from: handle, escalate_manager, escalate_team, archive.\n"
            "If routing is escalate_team, also pick a team from: engineering, legal, finance, hr.\n"
            "Otherwise set team to none.\n"
            "Rules: archive = spam/irrelevant. escalate_manager = needs executive awareness. "
            "escalate_team = needs a specific team. handle = you can deal with it yourself.\n"
            "Respond ONLY with valid JSON: {\"routing\": \"...\", \"team\": \"...\"}"
        ),
    }
    return (
        f"You are an email triage assistant.\n\n"
        f"EMAIL:\n"
        f"Subject:  {email.get('subject', '')}\n"
        f"From:     {email.get('sender', '')}\n"
        f"Received: {email.get('timestamp', '')}\n"
        f"Body:\n{email.get('body', '')}\n\n"
        f"TASK: {task_instructions[task_id]}"
    )


DEFAULT_ACTION = {"category": "normal", "priority": 3, "reply": "", "routing": "handle", "team": "none"}


def call_llm(client: OpenAI, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return ""


def parse_response(text: str) -> dict:
    try:
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # Fix unescaped real newlines inside JSON string values
        # e.g. "reply": "Dear Sir,\n\nThank you" — the \n here are actual newlines
        text = re.sub(
            r'(?<=: ")(.*?)(?="[\s,\}])',
            lambda m: m.group(0).replace('\n', '\\n').replace('\r', ''),
            text,
            flags=re.DOTALL,
        )

        data = json.loads(text)
        return {
            "category": str(data.get("category", "normal")).lower().strip(),
            "priority": int(data.get("priority", 3)),
            "reply":    str(data.get("reply", "")),
            "routing":  str(data.get("routing", "handle")).lower().strip(),
            "team":     str(data.get("team", "none")).lower().strip(),
        }
    except Exception:
        print(f"[DEBUG] parse failed, using defaults. raw={text[:80]}", flush=True)
        return DEFAULT_ACTION.copy()


# ---------------------------------------------------------------------------
# One episode — returns (rewards_list, steps_taken)
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: str):
    result  = env_reset()
    obs     = result.get("observation", {})
    rewards = []
    step    = 0

    while step < MAX_STEPS_PER_EP:
        email = obs.get("current_email")
        done  = result.get("done", False) or obs.get("done", False)

        if done or not email:
            if not rewards:
                log_step(step=1, action="no-op", reward=0.00, done=True, error="env returned done immediately")
            break

        step += 1
        prompt   = build_prompt(email, task_id)
        llm_text = call_llm(client, prompt)
        action   = parse_response(llm_text) if llm_text else DEFAULT_ACTION.copy()

        result = env_step(action["category"], action["priority"], action["reply"],
                          action.get("routing", "handle"), action.get("team", "none"))
        reward = float(result.get("reward") or 0.0)
        done   = result.get("done", False)
        obs    = result.get("observation", {})
        error  = obs.get("feedback", None) if reward == 0 else None

        action_str = (
            f"cat={action['category']} pri={action['priority']} "
            f"route={action['routing']} team={action['team']}"
        )
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        rewards.append(reward)

        if done or obs.get("done"):
            break

    return rewards, step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        log_start(task="classify", env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        raise SystemExit("ERROR: HF_TOKEN not set.")

    print(f"[DEBUG] LLM:  {API_BASE_URL} / {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Env:  {ENV_BASE_URL}", flush=True)

    client     = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    all_scores = {}

    for task_id in TASKS:
        episode_scores = []

        for ep in range(EPISODES_PER_TASK):
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            rewards, steps_taken = [], 0
            success = False
            score   = 0.0

            try:
                rewards, steps_taken = run_episode(client, task_id)
                score   = min(sum(rewards) / 5.0, 1.0)
                success = score > 0.0
            except Exception as e:
                print(f"[DEBUG] Episode crashed: {e}", flush=True)
                log_step(step=1, action="crash", reward=0.00, done=True, error=str(e))

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            episode_scores.append(score)

        avg = sum(episode_scores) / len(episode_scores)
        all_scores[task_id] = round(avg, 3)

    print("[DEBUG] === FINAL SCORES ===", flush=True)
    for task_id, score in all_scores.items():
        print(f"[DEBUG]   {task_id}: {score}", flush=True)


if __name__ == "__main__":
    main()