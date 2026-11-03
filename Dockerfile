FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .
RUN pip --default-timeout=300 --retries=10 install --no-cache-dir -r requirements-api.txt

COPY main.py ./main.py
COPY api/ ./api/
COPY deployment/ ./deployment/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]