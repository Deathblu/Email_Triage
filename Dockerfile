FROM python:3.11-bookworm

WORKDIR /app

# 1. Copy requirements from the server folder
# The path is relative to the Dockerfile (Root -> server -> requirements.txt)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy root level files
COPY models.py .
COPY pyproject.toml .
COPY openenv.yaml .

# 3. Copy the entire server folder to /app/server
# This ensures app.py and email_environment.py are exactly where they need to be
COPY server/ ./server/

ENV PYTHONUNBUFFERED=1
# This is crucial so uvicorn can find the 'server' module
ENV PYTHONPATH=/app

EXPOSE 7860

# Double check that server/app.py has an object named 'app'
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]