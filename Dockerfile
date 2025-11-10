# ===== Stage 1: Builder =====
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY . .

# ===== Stage 2: Runtime =====
FROM python:3.11-slim AS runtime

# Install only runtime library needed for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8000 8050
CMD ["bash"]

