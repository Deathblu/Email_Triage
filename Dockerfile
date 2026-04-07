FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (from server/ folder — same dir as this Dockerfile)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files from root context
COPY models.py ./models.py
COPY server/app.py ./server/app.py
COPY server/email_environment.py ./server/email_environment.py
COPY server/__init__.py ./server/__init__.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]