# ===== Base Image =====
FROM python:3.11-slim

# ===== System dependencies for LightGBM =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===== Set Working Directory =====
WORKDIR /app

# ===== Copy Project Files =====
COPY FINAL/requirements.txt /app/requirements.txt
COPY FINAL /app/FINAL

# ===== Install Python Dependencies =====
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# ===== Expose API Port =====
EXPOSE 8000

# ===== Launch the API =====
CMD ["uvicorn", "FINAL.api:app", "--host", "0.0.0.0", "--port", "8000"]
