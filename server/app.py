# server/app.py
#
# Wires EmailTriageEnvironment into a FastAPI server.
# We pass a factory function (lambda) to create_app so OpenEnv
# uses ONE shared instance across all requests — preserving state
# between reset() and step() calls.

from openenv.core.env_server import create_app
from server.email_environment import EmailTriageEnvironment
from models import EmailAction, EmailObservation

# Create ONE shared environment instance
_env_instance = EmailTriageEnvironment()

# Pass a factory that always returns the SAME instance
# This way reset() and step() share the same inbox state
app = create_app(lambda: _env_instance, EmailAction, EmailObservation)