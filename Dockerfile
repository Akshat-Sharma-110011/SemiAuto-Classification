FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y git libgomp1 && apt-get clean

COPY requirements-docker.txt .
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

COPY . .

EXPOSE 8020

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8020", "--reload"]
