FROM python:3.11-slim

WORKDIR /app

# PostgreSQL client ekle
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# startup.py'yi çalıştır
CMD ["python", "startup.py"]