FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
