FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/verify_pair.py scripts/verify_pair.py
COPY scripts/profile_system.py scripts/profile_system.py
COPY configs/ configs/

ENTRYPOINT ["python", "scripts/verify_pair.py"]
