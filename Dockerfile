FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip FIRST (critical)
RUN python -m pip install --upgrade pip

# Install PyTorch CPU wheels explicitly (prevents networkx issues)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy dummy model (for demo)
COPY results/checkpoints/model.pth results/checkpoints/model.pth

EXPOSE 8080

CMD ["uvicorn", "src.deployment.infer_api:app", "--host", "0.0.0.0", "--port", "8080"]

