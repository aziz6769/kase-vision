FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Apply the production UI repair before starting FastAPI.
RUN python3 fix_local.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
