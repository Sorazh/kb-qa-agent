FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r backend/requirements.txt

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
