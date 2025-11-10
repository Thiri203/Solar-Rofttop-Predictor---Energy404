# ===== Base Image =====
FROM python:3.11-slim

# ===== Set Working Directory =====
WORKDIR /app

# ===== System dependencies for LightGBM / XGBoost =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    git \
    gcc \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===== Copy Project Files =====
COPY . .

# ===== Install Python Dependencies =====
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# We will run two different commands from docker-compose, so no fixed CMD here
# (docker-compose will override command per service)
EXPOSE 8000 8050

CMD ["bash"]

# this is testing for github push not rewritting

