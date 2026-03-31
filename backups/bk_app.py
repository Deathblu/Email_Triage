# server/app.py
#
# This file wires our EmailTriageEnvironment into a FastAPI web server.
# It's deliberately short — OpenEnv's create_app() handles all the HTTP
# and WebSocket routes. We just hand it our environment and our types.
#
# When this file runs:
#   GET  /reset  → calls environment.reset(), returns EmailObservation as JSON
#   POST /step   → calls environment.step(action), returns EmailObservation as JSON
#   GET  /state  → calls environment.state(), returns EmailState as JSON
#   GET  /web    → opens the OpenEnv browser UI (for manual testing)

from openenv.core.env_server import create_app

# Import our environment logic
from .email_environment import EmailTriageEnvironment

# Import the types so create_app knows how to serialize/deserialize
from ..models import EmailAction, EmailObservation

# Instantiate the environment — one instance serves all requests.
# OpenEnv handles concurrency; our environment just needs to be stateless
# between episodes (which reset() ensures).
env = EmailTriageEnvironment()

# create_app() does all the work:
#   - Creates a FastAPI app
#   - Registers /reset, /step, /state routes
#   - Sets up WebSocket endpoint at /ws
#   - Registers the OpenEnv web UI at /web
app = create_app(env, EmailAction, EmailObservation)

# That's it. Run with:
#   uvicorn server.app:app --host 0.0.0.0 --port 8000
# Or inside Docker, the CMD in the Dockerfile handles this.
