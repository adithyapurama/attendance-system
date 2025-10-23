# --------------------------------------------------
# 🧱 Base Image (Debian Bullseye = Stable OpenCV)
# --------------------------------------------------
FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --------------------------------------------------
# ⚙️ System Dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-dev \
    libsm6 \
    libxext6 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# 🧩 Python Dependencies
# --------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=200 --retries 5 -r requirements.txt

# --------------------------------------------------
# 📂 Application Code
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# 🌐 Expose Port
# --------------------------------------------------
EXPOSE 5000

# --------------------------------------------------
# 🚀 Entrypoint
# --------------------------------------------------
ENTRYPOINT ["python", "main.py"]
CMD ["registration"]
